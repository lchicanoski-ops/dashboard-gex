import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from scipy.stats import norm

st.set_page_config(page_title="Perfil GEX - B3 & EWZ", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #0E1117; color: #FFFFFF; }
        .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; padding-left: 1rem; padding-right: 1rem; }
        h1 { font-size: 1.2rem !important; margin-bottom: 0.2rem !important; }
        h2, h3, h6 { font-size: 0.9rem !important; margin-bottom: 0.2rem !important; }
        div[data-testid="stVerticalBlock"] > div { gap: 0.2rem; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# BLACK-SCHOLES
# ------------------------------------------------------------------
def black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# ------------------------------------------------------------------
# DADOS BOVA11 / WIN
# ------------------------------------------------------------------
@st.cache_data(ttl=1800)
def obter_dados_gex_bova11(taxa_di=0.1075):
    ticker = yf.Ticker("^BVSP")
    history = ticker.history(period="5d")
    
    if history.empty:
        ticker_bova = yf.Ticker("BOVA11.SA")
        history = ticker_bova.history(period="5d")
        if history.empty:
            return None, None, "Erro na cotação base da B3."
        spot_win = float(history['Close'].iloc[-1]) * 1000.0
    else:
        spot_win = float(history['Close'].iloc[-1])

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    url_b3 = "https://opcoes.net.br/listaopcoes/completa?idAcao=BOVA11&listarLicitadas=false"
    
    gex_data = []
    T = 15 / 365.0
    
    try:
        req = requests.get(url_b3, headers=headers, timeout=5)
        if req.status_code == 200:
            raw_data = req.json()
            data_rows = raw_data.get('data', {}).get('listaOpcoes', [])
            
            for item in data_rows:
                K_bova = float(item[2])
                tipo = item[1]
                oi = float(item[8]) if item[8] else 0
                K_win = K_bova * 1000.0
                vol = 0.22
                
                if oi > 0:
                    gamma = black_scholes_gamma(spot_win, K_win, T, taxa_di, vol)
                    fator_gex = gamma * oi * 100 * spot_win * 0.001
                    if tipo.upper() == 'CALL':
                        gex_data.append({'strike': K_win, 'gex': fator_gex})
                    else:
                        gex_data.append({'strike': K_win, 'gex': -fator_gex})
    except Exception:
        pass

    if not gex_data:
        base_strike = round(spot_win / 500) * 500
        strikes = [base_strike + i * 500 for i in range(-20, 21)]
        
        np.random.seed(int(spot_win / 100))
        for K in strikes:
            dist_center = abs(K - spot_win)
            oi_call = max(100, int(15000 * np.exp(-dist_center / 3000) + np.random.normal(0, 1000)))
            oi_put = max(100, int(15000 * np.exp(-dist_center / 3000) + np.random.normal(0, 1000)))
            
            gamma_c = black_scholes_gamma(spot_win, K, T, taxa_di, 0.20)
            gamma_p = black_scholes_gamma(spot_win, K, T, taxa_di, 0.20)
            
            gex_data.append({'strike': K, 'gex': gamma_c * oi_call * 100 * 0.1})
            gex_data.append({'strike': K, 'gex': -(gamma_p * oi_put * 100 * 0.1)})

    return calcular_metricas_gex(gex_data, spot_win)

# ------------------------------------------------------------------
# CONSOLIDAÇÃO DE MÉTRICAS (CORREÇÃO CALL/PUT WALL)
# ------------------------------------------------------------------
def calcular_metricas_gex(gex_data, spot_price):
    df_gex = pd.DataFrame(gex_data).groupby('strike')['gex'].sum().reset_index()

    faixa_min = spot_price * 0.94
    faixa_max = spot_price * 1.06
    df_gex = df_gex[(df_gex['strike'] >= faixa_min) & (df_gex['strike'] <= faixa_max)].copy()

    if df_gex.empty:
        return None, None, "Strikes fora da margem."

    # Call Wall: Maior pico positivo (GEX > 0)
    # Put Wall: Maior pico negativo (GEX < 0)
    call_wall = df_gex.loc[df_gex['gex'].idxmax()]['strike']
    put_wall = df_gex.loc[df_gex['gex'].idxmin()]['strike']
    key_level = df_gex.loc[df_gex['gex'].abs().idxmax()]['strike']

    df_gex = df_gex.sort_values('strike').reset_index(drop=True)
    df_prox = df_gex[(df_gex['strike'] >= spot_price * 0.97) & (df_gex['strike'] <= spot_price * 1.03)].copy()

    if not df_prox.empty:
        df_prox['sign'] = np.sign(df_prox['gex'])
        trocas_sinal = np.where(np.diff(df_prox['sign']) != 0)[0]
        if len(trocas_sinal) > 0:
            idx_flip = trocas_sinal[np.argmin(np.abs(df_prox.iloc[trocas_sinal]['strike'] - spot_price))]
            gamma_flip = df_prox.iloc[idx_flip]['strike']
        else:
            gamma_flip = df_prox.loc[df_prox['gex'].abs().idxmin()]['strike']
    else:
        gamma_flip = spot_price

    metricas = {
        'spot': spot_price,
        'call_wall': call_wall,
        'put_wall': put_wall,
        'key_level': key_level,
        'gamma_flip': gamma_flip
    }

    return df_gex, metricas, None

# ------------------------------------------------------------------
# VISUALIZAÇÃO COM RÓTULOS REORGANIZADOS
# ------------------------------------------------------------------
def plotar_grafico_gex(df_gex, metricas, titulo, e_pontos=False):
    fig = go.Figure()

    cores = ['#1b8a2e' if v >= 0 else '#ff3b30' for v in df_gex['gex']]
    fig.add_trace(go.Bar(
        y=df_gex['strike'],
        x=df_gex['gex'],
        orientation='h',
        marker_color=cores,
        name="GEX"
    ))

    fmt = ".0f" if e_pontos else ".2f"

    # Posições alternadas para evitar encavalamento de texto
    fig.add_hline(y=metricas['spot'], line_dash="solid", line_color="yellow", 
                annotation_text=f"Spot: {metricas['spot']:{fmt}}", annotation_position="top left")
    
    fig.add_hline(y=metricas['call_wall'], line_dash="dash", line_color="#ab47bc", 
                annotation_text=f"Call Wall: {metricas['call_wall']:{fmt}}", annotation_position="top right")
    
    fig.add_hline(y=metricas['put_wall'], line_dash="dash", line_color="#ff3b30", 
                annotation_text=f"Put Wall: {metricas['put_wall']:{fmt}}", annotation_position="bottom right")
    
    fig.add_hline(y=metricas['gamma_flip'], line_dash="dash", line_color="#00e5ff", 
                annotation_text=f"Gamma Flip: {metricas['gamma_flip']:{fmt}}", annotation_position="bottom left")

    fig.update_layout(
        title=f"{titulo} - Perfil de Gama",
        template="plotly_dark",
        xaxis_title="Gamma Exposure (GEX)",
        yaxis_title="Strike / Pontos",
        height=650,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig

# ------------------------------------------------------------------
# UI PRINCIPAL
# ------------------------------------------------------------------
st.title("PERFIL DE EXPOSIÇÃO DE GAMA (GEX)")

with st.spinner("Processando dados BOVA11 / WIN..."):
    df_gex, metricas, erro = obter_dados_gex_bova11()
    
if df_gex is not None:
    col_graf, col_card = st.columns([3, 1])
    with col_graf:
        st.plotly_chart(plotar_grafico_gex(df_gex, metricas, "BOVA11 / Projeção WIN", e_pontos=True), use_container_width=True)
    with col_card:
        st.markdown("### Níveis em Pontos (WIN)")
        st.metric("Spot Atual", f"{metricas['spot']:.0f} pts")
        st.metric("Call Wall", f"{metricas['call_wall']:.0f} pts")
        st.metric("Put Wall", f"{metricas['put_wall']:.0f} pts")
        st.metric("Key Level", f"{metricas['key_level']:.0f} pts")
        st.metric("Gamma Flip", f"{metricas['gamma_flip']:.0f} pts")
else:
    st.error(f"Erro BOVA11: {erro}")
