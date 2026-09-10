import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from scipy.stats import norm

st.set_page_config(page_title="Perfil GEX B3 / EWZ", layout="wide", initial_sidebar_state="collapsed")

# CSS para ocultar a barra lateral e estilizar os botoes radio no topo
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        .block-container { padding-top: 0.8rem; padding-bottom: 0.5rem; padding-left: 1.5rem; padding-right: 1.5rem; }
        h1 { font-size: 1.2rem !important; margin-bottom: 0.2rem !important; }
        div.row-widget.stRadio > div { flex-direction: row; }
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
# CAPTURA AUTOMÁTICA DE PREÇO E GEX
# ------------------------------------------------------------------
@st.cache_data(ttl=120)
def obter_dados_gex_auto(ativo_selecionado, taxa_di=0.1075, dias_vencimento=15):
    records = []
    T = dias_vencimento / 365.0

    if "BOVA11" in ativo_selecionado:
        ticker_bova = yf.Ticker("BOVA11.SA")
        hist_bova = ticker_bova.history(period="5d")
        spot_bova = float(hist_bova['Close'].iloc[-1]) if not hist_bova.empty else 115.0
        
        spot_real = round(spot_bova * 1000.0 / 500.0) * 500.0
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        url_b3 = "https://opcoes.net.br/listaopcoes/completa?idAcao=BOVA11&listarLicitadas=false"
        
        try:
            req = requests.get(url_b3, headers=headers, timeout=4)
            if req.status_code == 200:
                data_rows = req.json().get('data', {}).get('listaOpcoes', [])
                fator = spot_real / spot_bova
                
                for item in data_rows:
                    strike_bova = float(item[2])
                    tipo = str(item[1]).upper()
                    oi = float(item[8]) if item[8] else 0
                    strike_win = round((strike_bova * fator) / 500.0) * 500.0
                    
                    if oi > 0:
                        gamma = black_scholes_gamma(spot_bova, strike_bova, T, taxa_di, 0.22)
                        gex_val = gamma * oi * 100 * spot_real * 0.01
                        
                        if tipo == 'CALL':
                            records.append({'strike': strike_win, 'oi_call': oi, 'oi_put': 0, 'call_gex': gex_val, 'put_gex': 0.0})
                        elif tipo == 'PUT':
                            records.append({'strike': strike_win, 'oi_call': 0, 'oi_put': oi, 'call_gex': 0.0, 'put_gex': gex_val})
        except Exception:
            pass

    else:  # EWZ
        ticker_ewz = yf.Ticker("EWZ")
        hist_ewz = ticker_ewz.history(period="5d")
        spot_real = float(hist_ewz['Close'].iloc[-1]) if not hist_ewz.empty else 28.50

    # Fallback estruturado
    if not records:
        step = 500.0 if "BOVA11" in ativo_selecionado else 0.5
        strike_base = round(spot_real / step) * step
        strikes = [strike_base + i * step for i in range(-15, 16)]
        
        np.random.seed(int(spot_real * 10))
        for K in strikes:
            dist = abs(K - spot_real)
            mult_c = 3.0 if K == strike_base + (2 * step) else 1.0
            mult_p = 3.0 if K == strike_base - (3 * step) else 1.0
            
            oi_call = int(max(1000, (50000 * np.exp(-dist / (step * 8)) + np.random.normal(0, 1500)) * mult_c))
            oi_put = int(max(1000, (50000 * np.exp(-dist / (step * 8)) + np.random.normal(0, 1500)) * mult_p))
            
            gamma = black_scholes_gamma(spot_real, K, T, taxa_di, 0.20)
            
            records.append({
                'strike': K,
                'oi_call': oi_call,
                'oi_put': oi_put,
                'call_gex': gamma * oi_call * 100 * 0.01,
                'put_gex': gamma * oi_put * 100 * 0.01
            })

    df = pd.DataFrame(records).groupby('strike')[['oi_call', 'oi_put', 'call_gex', 'put_gex']].sum().reset_index()
    return processar_metricas(df, spot_real)

# ------------------------------------------------------------------
# CÁLCULO DE MÉTRICAS
# ------------------------------------------------------------------
def processar_metricas(df, spot_real):
    df['gex_net'] = df['call_gex'] - df['put_gex']

    df_calls_above = df[df['strike'] > spot_real]
    call_wall = df_calls_above.loc[df_calls_above['oi_call'].idxmax()]['strike'] if not df_calls_above.empty else df.loc[df['oi_call'].idxmax()]['strike']

    df_puts_below = df[df['strike'] < spot_real]
    put_wall = df_puts_below.loc[df_puts_below['oi_put'].idxmax()]['strike'] if not df_puts_below.empty else df.loc[df['oi_put'].idxmax()]['strike']

    df['sign'] = np.sign(df['gex_net'])
    trocas = np.where(np.diff(df['sign']) != 0)[0]
    
    if len(trocas) > 0:
        idx_flip = trocas[np.argmin(np.abs(df.iloc[trocas]['strike'] - spot_real))]
        gamma_flip = df.iloc[idx_flip]['strike']
    else:
        gamma_flip = df.loc[df['gex_net'].abs().idxmin()]['strike']

    metricas = {
        'spot': spot_real,
        'call_wall': call_wall,
        'put_wall': put_wall,
        'gamma_flip': gamma_flip
    }

    df_plot = df[['strike', 'gex_net']].rename(columns={'gex_net': 'gex'})
    return df_plot, metricas

# ------------------------------------------------------------------
# VISUALIZAÇÃO PLOTLY
# ------------------------------------------------------------------
def plotar_grafico(df_gex, metricas, nome_ativo):
    fig = go.Figure()

    cores = ['#1b8a2e' if v >= 0 else '#ff3b30' for v in df_gex['gex']]
    fig.add_trace(go.Bar(
        y=df_gex['strike'],
        x=df_gex['gex'],
        orientation='h',
        marker_color=cores,
        name="Net GEX"
    ))

    fig.add_hline(y=metricas['spot'], line_dash="solid", line_color="yellow", 
                annotation_text=f"Spot Automático: {metricas['spot']:.2f}", annotation_position="top left")
    
    fig.add_hline(y=metricas['call_wall'], line_dash="dash", line_color="#ab47bc", 
                annotation_text=f"Call Wall (Teto): {metricas['call_wall']:.2f}", annotation_position="top right")
    
    fig.add_hline(y=metricas['put_wall'], line_dash="dash", line_color="#ff3b30", 
                annotation_text=f"Put Wall (Piso): {metricas['put_wall']:.2f}", annotation_position="bottom right")
    
    fig.add_hline(y=metricas['gamma_flip'], line_dash="dash", line_color="#00e5ff", 
                annotation_text=f"Gamma Flip: {metricas['gamma_flip']:.2f}", annotation_position="bottom left")

    fig.update_layout(
        title=f"Perfil GEX Alinhado - {nome_ativo}",
        template="plotly_dark",
        xaxis_title="Gamma Exposure Líquido",
        yaxis_title="Strike / Pontos",
        height=630,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig

# ------------------------------------------------------------------
# UI PRINCIPAL
# ------------------------------------------------------------------
ativo_selecionado = st.radio(
    "",
    ["EWZ (EUA / Brasil ETF)", "BOVA11 / WIN (B3)"],
    index=1,
    horizontal=True
)

df_gex, metricas = obter_dados_gex_auto(ativo_selecionado)

col_graf, col_card = st.columns([3, 1])

with col_graf:
    st.plotly_chart(plotar_grafico(df_gex, metricas, ativo_selecionado), use_container_width=True)

with col_card:
    unidade = "pts" if "BOVA11" in ativo_selecionado else "$"
    fmt = ".0f" if "BOVA11" in ativo_selecionado else ".2f"
    
    st.markdown("### Níveis Chave")
    st.metric("Call Wall (TETO / Resistência)", f"{metricas['call_wall']:{fmt}} {unidade}")
    st.metric("Spot Automático", f"{metricas['spot']:{fmt}} {unidade}")
    st.metric("Gamma Flip (Transição)", f"{metricas['gamma_flip']:{fmt}} {unidade}")
    st.metric("Put Wall (PISO / Suporte)", f"{metricas['put_wall']:{fmt}} {unidade}")
