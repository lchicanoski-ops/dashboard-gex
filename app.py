import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Perfil GEX - Gamma Exposure", layout="wide")

# CSS para estilo escuro e denso
st.markdown("""
    <style>
        .stApp { background-color: #0E1117; color: #FFFFFF; }
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# MOTOR DE CÁLCULO GEX (BLACK-SCHOLES)
# ------------------------------------------------------------------
def black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

@st.cache_data(ttl=300)
def obter_dados_gex(ticker_symbol, taxa_juros=0.045, multiplicador_carrego=1.0):
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period="5d")
    
    if history.empty:
        return None, None
        
    spot_price = history['Close'].iloc[-1] * multiplicador_carrego
    expirations = ticker.options
    
    if not expirations:
        return None, None
    
    exp_date = expirations[0]  # Primeiro vencimento (curto prazo)
    opt_chain = ticker.option_chain(exp_date)
    
    calls = opt_chain.calls[['strike', 'openInterest', 'impliedVolatility']].dropna()
    puts = opt_chain.puts[['strike', 'openInterest', 'impliedVolatility']].dropna()
    
    T = 15 / 365.0
    gex_data = []

    # Calls (Gama Positivo)
    for _, row in calls.iterrows():
        K = row['strike'] * multiplicador_carrego
        oi = row['openInterest']
        vol = row['impliedVolatility']
        if vol > 0 and oi > 0:
            gamma = black_scholes_gamma(spot_price, K, T, taxa_juros, vol)
            gex = gamma * oi * 100 * spot_price * 0.01
            gex_data.append({'strike': K, 'gex': gex})

    # Puts (Gama Negativo)
    for _, row in puts.iterrows():
        K = row['strike'] * multiplicador_carrego
        oi = row['openInterest']
        vol = row['impliedVolatility']
        if vol > 0 and oi > 0:
            gamma = black_scholes_gamma(spot_price, K, T, taxa_juros, vol)
            gex = -(gamma * oi * 100 * spot_price * 0.01)
            gex_data.append({'strike': K, 'gex': gex})

    if not gex_data:
        return None, None

    df_gex = pd.DataFrame(gex_data).groupby('strike')['gex'].sum().reset_index()
    
    # Filtrar apenas strikes relevantes ao redor do Spot (evita distorção de pozinhos)
    faixa_min = spot_price * 0.85
    faixa_max = spot_price * 1.15
    df_gex = df_gex[(df_gex['strike'] >= faixa_min) & (df_gex['strike'] <= faixa_max)].copy()

    if df_gex.empty:
        return None, None

    # Níveis Chave
    call_wall = df_gex.loc[df_gex['gex'].idxmax()]['strike']
    put_wall = df_gex.loc[df_gex['gex'].idxmin()]['strike']
    key_level = df_gex.loc[df_gex['gex'].abs().idxmax()]['strike']
    
    # Cálculo do Gamma Flip próximo ao Spot
    df_gex = df_gex.sort_values('strike').reset_index(drop=True)
    df_gex['cumsum'] = df_gex['gex'].cumsum()
    
    zero_crossings = np.where(np.diff(np.sign(df_gex['cumsum'])))[0]
    
    if len(zero_crossings) > 0:
        idx_prox = zero_crossings[np.argmin(np.abs(df_gex.loc[zero_crossings, 'strike'] - spot_price))]
        gamma_flip = df_gex.iloc[idx_prox]['strike']
    else:
        gamma_flip = spot_price

    metricas = {
        'spot': spot_price,
        'call_wall': call_wall,
        'put_wall': put_wall,
        'key_level': key_level,
        'gamma_flip': gamma_flip,
        'max_gamma': df_gex['strike'].max(),
        'min_gamma': df_gex['strike'].min()
    }

    return df_gex, metricas

# ------------------------------------------------------------------
# DESENHO DO GRÁFICO SPOTGAMMA
# ------------------------------------------------------------------
def plotar_grafico_gex(df_gex, metricas, titulo):
    fig = go.Figure()

    cores = ['#1b8a2e' if v >= 0 else '#ff3b30' for v in df_gex['gex']]
    fig.add_trace(go.Bar(
        y=df_gex['strike'],
        x=df_gex['gex'],
        orientation='h',
        marker_color=cores,
        name="Gamma Exposure"
    ))

    # Adicionando Linhas dos Níveis Chave
    fig.add_hline(y=metricas['spot'], line_dash="solid", line_color="yellow", annotation_text=f"Spot Price: {metricas['spot']:.2f}")
    fig.add_hline(y=metricas['call_wall'], line_dash="dash", line_color="#ab47bc", annotation_text=f"Call Wall: {metricas['call_wall']:.2f}")
    fig.add_hline(y=metricas['put_wall'], line_dash="dash", line_color="#ff3b30", annotation_text=f"Put Wall: {metricas['put_wall']:.2f}")
    fig.add_hline(y=metricas['gamma_flip'], line_dash="dash", line_color="#00e5ff", annotation_text=f"Gamma Flip: {metricas['gamma_flip']:.2f}")

    fig.update_layout(
        title=f"{titulo} - Perfil de Gama por Strike",
        template="plotly_dark",
        xaxis_title="Spot Gamma Exposure",
        yaxis_title="Strike",
        height=680,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig

# ------------------------------------------------------------------
# INTERFACE PRINCIPAL
# ------------------------------------------------------------------
st.title("PERFIL DE EXPOSIÇÃO DE GAMA (GEX)")

col_sel, col_info = st.columns([1, 3])

with col_sel:
    ativo = st.radio("Selecione o Mercado", ["EWZ (EUA / Brasil ETF)", "WIN / BOVA11 (B3)"])

if ativo == "EWZ (EUA / Brasil ETF)":
    with st.spinner("Buscando cadeia de opções do EWZ..."):
        df_gex, metricas = obter_dados_gex("EWZ", taxa_juros=0.045, multiplicador_carrego=1.0)
    
    if df_gex is not None:
        col_graf, col_card = st.columns([3, 1])
        with col_graf:
            st.plotly_chart(plotar_grafico_gex(df_gex, metricas, "EWZ"), use_container_width=True)
        with col_card:
            st.markdown("### Níveis Chave")
            st.metric("Spot Price", f"${metricas['spot']:.2f}")
            st.metric("Call Wall", f"${metricas['call_wall']:.2f}")
            st.metric("Put Wall", f"${metricas['put_wall']:.2f}")
            st.metric("Key Level", f"${metricas['key_level']:.2f}")
            st.metric("Gamma Flip", f"${metricas['gamma_flip']:.2f}")
    else:
        st.error("Erro ao carregar dados do EWZ.")

else:
    with st.spinner("Buscando dados de BOVA11 e calculando carrego..."):
        taxa_di = 0.1075  # Taxa DI/Selic
        fator_win = 1000 * (1 + (taxa_di * (15 / 365)))  # Fator de conversão do contrato
        df_gex, metricas = obter_dados_gex("BOVA11.SA", taxa_juros=taxa_di, multiplicador_carrego=fator_win)
        
    if df_gex is not None:
        col_graf, col_card = st.columns([3, 1])
        with col_graf:
            st.plotly_chart(plotar_grafico_gex(df_gex, metricas, "WIN / BOVA11"), use_container_width=True)
        with col_card:
            st.markdown("### Níveis Chave")
            st.metric("Spot Convertido", f"{metricas['spot']:.0f} pts")
            st.metric("Call Wall", f"{metricas['call_wall']:.0f} pts")
            st.metric("Put Wall", f"{metricas['put_wall']:.0f} pts")
            st.metric("Key Level", f"{metricas['key_level']:.0f} pts")
            st.metric("Gamma Flip", f"{metricas['gamma_flip']:.0f} pts")
    else:
        st.error("Erro ao carregar dados do BOVA11.")
