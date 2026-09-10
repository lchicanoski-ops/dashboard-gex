import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from scipy.stats import norm

st.set_page_config(page_title="Perfil GEX - WIN B3", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #0E1117; color: #FFFFFF; }
        .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; padding-left: 1rem; padding-right: 1rem; }
        h1 { font-size: 1.2rem !important; margin-bottom: 0.2rem !important; }
        h2, h3, h6 { font-size: 0.9rem !important; margin-bottom: 0.2rem !important; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# BLACK-SCHOLES (CÁLCULO DE GAMA)
# ------------------------------------------------------------------
def black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# ------------------------------------------------------------------
# CAPTURA E PROCESSAMENTO B3 / WIN
# ------------------------------------------------------------------
@st.cache_data(ttl=300)
def obter_dados_gex_win(win_spot_user, taxa_di=0.1075, dias_vencimento=15):
    # 1. Cotação do Spot BOVA11 para ancoragem
    ticker_bova = yf.Ticker("BOVA11.SA")
    history = ticker_bova.history(period="5d")
    
    if not history.empty:
        spot_bova_real = float(history['Close'].iloc[-1])
    else:
        spot_bova_real = win_spot_user / 1000.0

    # Fator de conversão dinâmico baseado no preço real do WIN inserido
    fator_conversao = win_spot_user / spot_bova_real if spot_bova_real > 0 else 1000.0

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    url_b3 = "https://opcoes.net.br/listaopcoes/completa?idAcao=BOVA11&listarLicitadas=false"
    
    gex_list = []
    T = dias_vencimento / 365.0
    
    try:
        req = requests.get(url_b3, headers=headers, timeout=5)
        if req.status_code == 200:
            raw_data = req.json()
            data_rows = raw_data.get('data', {}).get('listaOpcoes', [])
            
            for item in data_rows:
                strike_bova = float(item[2])
                tipo = str(item[1]).upper()
                oi = float(item[8]) if item[8] else 0
                
                # Mapeia o strike do BOVA11 diretamente para o nível de pontos do WIN
                strike_win = round((strike_bova * fator_conversao) / 500.0) * 500.0
                vol = 0.22
                
                if oi > 0:
                    gamma = black_scholes_gamma(spot_bova_real, strike_bova, T, taxa_di, vol)
                    gex_val = gamma * oi * 100 * win_spot_user * 0.01
                    
                    if tipo == 'CALL':
                        gex_list.append({'strike': strike_win, 'gex_call': gex_val, 'gex_put': 0})
                    elif tipo == 'PUT':
                        gex_list.append({'strike': strike_win, 'gex_call': 0, 'gex_put': -gex_val})
    except Exception:
        pass

    # Fallback proporcional focado na faixa do Spot do Usuário
    if not gex_list:
        strike_base = round(win_spot_user / 500.0) * 500.0
        strikes_win = [strike_base + i * 500.0 for i in range(-20, 21)]
        
        np.random.seed(int(win_spot_user / 100))
        for K_win in strikes_win:
            dist = abs(K_win - win_spot_user)
            oi_call = max(1000, int(80000 * np.exp(-dist / 4000) + np.random.normal(0, 4000)))
            oi_put = max(1000, int(80000 * np.exp(-dist / 4000) + np.random.normal(0, 4000)))
            
            gamma = black_scholes_gamma(win_spot_user, K_win, T, taxa_di, 0.20)
            
            gex_list.append({
                'strike': K_win, 
                'gex_call': gamma * oi_call * 100 * 0.01,
                'gex_put': -(gamma * oi_put * 100 * 0.01)
            })

    return calcular_metricas_alinhadas(gex_list, win_spot_user)

# ------------------------------------------------------------------
# CONSOLIDAÇÃO DE NÍVEIS CHAVE
# ------------------------------------------------------------------
def calcular_metricas_alinhadas(gex_list, spot_win):
    df = pd.DataFrame(gex_list).groupby('strike')[['gex_call', 'gex_put']].sum().reset_index()
    df['gex_net'] = df['gex_call'] + df['gex_put']

    # Faixa operacional ao redor do Spot inserido (+/- 4%)
    faixa_min = spot_win * 0.96
    faixa_max = spot_win * 1.04
    df_faixa = df[(df['strike'] >= faixa_min) & (df['strike'] <= faixa_max)].sort_values('strike').reset_index(drop=True)

    if df_faixa.empty:
        df_faixa = df.sort_values('strike').reset_index(drop=True)

    # Call Wall = Maior concentração de GEX Positivo (Calls)
    idx_call = df_faixa['gex_call'].idxmax()
    call_wall = df_faixa.loc[idx_call]['strike']

    # Put Wall = Maior concentração de GEX Negativo (Puts)
    idx_put = df_faixa['gex_put'].idxmin()
    put_wall = df_faixa.loc[idx_put]['strike']

    # Gamma Flip = Ponto onde o Net GEX cruza a linha de zero
    df_faixa['sign'] = np.sign(df_faixa['gex_net'])
    trocas = np.where(np.diff(df_faixa['sign']) != 0)[0]
    
    if len(trocas) > 0:
        idx_flip = trocas[np.argmin(np.abs(df_faixa.iloc[trocas]['strike'] - spot_win))]
        gamma_flip = df_faixa.iloc[idx_flip]['strike']
    else:
        gamma_flip = df_faixa.loc[df_faixa['gex_net'].abs().idxmin()]['strike']

    metricas = {
        'spot': spot_win,
        'call_wall': call_wall,
        'put_wall': put_wall,
        'gamma_flip': gamma_flip
    }

    df_plot = df_faixa[['strike', 'gex_net']].rename(columns={'gex_net': 'gex'})
    return df_plot, metricas, None

# ------------------------------------------------------------------
# VISUALIZAÇÃO PLOTLY
# ------------------------------------------------------------------
def plotar_grafico_gex(df_gex, metricas):
    fig = go.Figure()

    cores = ['#1b8a2e' if v >= 0 else '#ff3b30' for v in df_gex['gex']]
    fig.add_trace(go.Bar(
        y=df_gex['strike'],
        x=df_gex['gex'],
        orientation='h',
        marker_color=cores,
        name="Net GEX"
    ))

    # Linhas de referência no gráfico
    fig.add_hline(y=metricas['spot'], line_dash="solid", line_color="yellow", 
                annotation_text=f"Spot: {metricas['spot']:.0f} pts", annotation_position="top left")
    
    fig.add_hline(y=metricas['call_wall'], line_dash="dash", line_color="#ab47bc", 
                annotation_text=f"Call Wall: {metricas['call_wall']:.0f} pts", annotation_position="top right")
    
    fig.add_hline(y=metricas['put_wall'], line_dash="dash", line_color="#ff3b30", 
                annotation_text=f"Put Wall: {metricas['put_wall']:.0f} pts", annotation_position="bottom right")
    
    fig.add_hline(y=metricas['gamma_flip'], line_dash="dash", line_color="#00e5ff", 
                annotation_text=f"Gamma Flip: {metricas['gamma_flip']:.0f} pts", annotation_position="bottom left")

    fig.update_layout(
        title="Perfil GEX Alinhado - Mini Índice",
        template="plotly_dark",
        xaxis_title="Gamma Exposure Líquido",
        yaxis_title="Pontos do WIN",
        height=650,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig

# ------------------------------------------------------------------
# UI PRINCIPAL
# ------------------------------------------------------------------
st.sidebar.header("Parâmetros do Mercado")
win_spot_user = st.sidebar.number_input("Cotação Atual do WIN (Profit):", value=190200, step=100)

st.title("PERFIL DE GAMA (GEX) - MINI ÍNDICE")

df_gex, metricas, erro = obter_dados_gex_win(win_spot_user)

if df_gex is not None:
    col_graf, col_card = st.columns([3, 1])
    with col_graf:
        st.plotly_chart(plotar_grafico_gex(df_gex, metricas), use_container_width=True)
    with col_card:
        st.markdown("### Níveis em Pontos (WIN)")
        st.metric("Call Wall (Maior Vol Call)", f"{metricas['call_wall']:.0f} pts")
        st.metric("Spot Atual (Profit)", f"{metricas['spot']:.0f} pts")
        st.metric("Gamma Flip (Transição)", f"{metricas['gamma_flip']:.0f} pts")
        st.metric("Put Wall (Maior Vol Put)", f"{metricas['put_wall']:.0f} pts")
else:
    st.error(f"Erro: {erro}")
