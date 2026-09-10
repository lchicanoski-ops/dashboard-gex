import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import requests

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
# MATEMÁTICA BLACK-SCHOLES
# ------------------------------------------------------------------
def black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# ------------------------------------------------------------------
# CAPTURA DE DADOS: EWZ (YAHOO FINANCE)
# ------------------------------------------------------------------
@st.cache_data(ttl=300)
def obter_dados_gex_ewz(taxa_juros=0.045):
    ticker = yf.Ticker("EWZ")
    
    spot_price = None
    try:
        spot_price = ticker.fast_info['lastPrice']
    except Exception:
        pass

    if spot_price is None or np.isnan(spot_price):
        history = ticker.history(period="5d", interval="5m", prepost=True)
        if not history.empty:
            spot_price = history['Close'].iloc[-1]

    if spot_price is None or np.isnan(spot_price):
        return None, None, "Falha ao obter preço Spot do EWZ."

    try:
        expirations = ticker.options
        if not expirations:
            return None, None, "Sem vencimentos para EWZ."
        
        opt_chain = ticker.option_chain(expirations[0])
        calls = opt_chain.calls[['strike', 'openInterest', 'impliedVolatility']].fillna(0)
        puts = opt_chain.puts[['strike', 'openInterest', 'impliedVolatility']].fillna(0)
    except Exception as e:
        return None, None, f"Erro ao acessar cadeia do EWZ: {e}"

    T = 15 / 365.0
    gex_data = []

    for _, row in calls.iterrows():
        K, oi, vol = row['strike'], row['openInterest'], row['impliedVolatility']
        if vol > 0 and oi > 0:
            gamma = black_scholes_gamma(spot_price, K, T, taxa_juros, vol)
            gex_data.append({'strike': K, 'gex': gamma * oi * 100 * spot_price * 0.01})

    for _, row in puts.iterrows():
        K, oi, vol = row['strike'], row['openInterest'], row['impliedVolatility']
        if vol > 0 and oi > 0:
            gamma = black_scholes_gamma(spot_price, K, T, taxa_juros, vol)
            gex_data.append({'strike': K, 'gex': -(gamma * oi * 100 * spot_price * 0.01)})

    return calcular_metricas_gex(gex_data, spot_price, 1.0)


# ------------------------------------------------------------------
# CAPTURA DE DADOS: BOVA11 (OPÇÕES.NET SCRAPING REQUISITADO)
# ------------------------------------------------------------------
@st.cache_data(ttl=300)
def obter_dados_gex_bova11(taxa_di=0.1075):
    # Projeção do Spot BOVA11 para Pontos do WIN
    fator_win = 1000 * (1 + (taxa_di * (15 / 365)))
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest'
    }
    
    url = "https://www.opcoes.net.br/opcao/getgrade/BOVA11"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        if not data.get('success'):
            return None, None, "Opções.net não retornou dados válidos para BOVA11."

        spot_price = float(data['data']['cotacao'])
        rows = data['data']['grid']
        
        gex_data = []
        T = 15 / 365.0

        for r in rows:
            # Estrutura padrão da matriz da grade do Opções.net
            strike = float(r.get('strike', 0))
            
            # Posição Aberta (Open Interest) de Calls e Puts
            oi_call = float(r.get('call_po', 0) or 0)
            oi_put = float(r.get('put_po', 0) or 0)
            
            # Volatilidade Implícita estimada/padrão B3
            vol_call = float(r.get('call_vi', 0) or 0.20) / 100.0 if float(r.get('call_vi', 0) or 0) > 0 else 0.20
            vol_put = float(r.get('put_vi', 0) or 0.20) / 100.0 if float(r.get('put_vi', 0) or 0) > 0 else 0.20

            if strike > 0:
                if oi_call > 0:
                    gamma_c = black_scholes_gamma(spot_price, strike, T, taxa_di, vol_call)
                    gex_data.append({'strike': strike * fator_win, 'gex': gamma_c * oi_call * 100 * spot_price * 0.01})

                if oi_put > 0:
                    gamma_p = black_scholes_gamma(spot_price, strike, T, taxa_di, vol_put)
                    gex_data.append({'strike': strike * fator_win, 'gex': -(gamma_p * oi_put * 100 * spot_price * 0.01)})

        return calcular_metricas_gex(gex_data, spot_price, fator_win)

    except Exception as e:
        return None, None, f"Erro ao conectar com Opções.net: {e}"


# ------------------------------------------------------------------
# CONSOLIDAÇÃO DE MÉTRICAS E FLIP
# ------------------------------------------------------------------
def calcular_metricas_gex(gex_data, spot_price, multiplicador):
    if not gex_data:
        return None, None, "Sem exposição computável."

    df_gex = pd.DataFrame(gex_data).groupby('strike')['gex'].sum().reset_index()
    spot_convertido = spot_price * multiplicador

    faixa_min = spot_convertido * 0.90
    faixa_max = spot_convertido * 1.10
    df_gex = df_gex[(df_gex['strike'] >= faixa_min) & (df_gex['strike'] <= faixa_max)].copy()

    if df_gex.empty:
        return None, None, "Strikes fora da margem operacional."

    call_wall = df_gex.loc[df_gex['gex'].idxmax()]['strike']
    put_wall = df_gex.loc[df_gex['gex'].idxmin()]['strike']
    key_level = df_gex.loc[df_gex['gex'].abs().idxmax()]['strike']

    # Gamma Flip Calibrado na Vizinhança Imediata
    df_gex = df_gex.sort_values('strike').reset_index(drop=True)
    df_prox = df_gex[(df_gex['strike'] >= spot_convertido * 0.94) & (df_gex['strike'] <= spot_convertido * 1.06)].copy()

    if not df_prox.empty:
        df_prox['sign'] = np.sign(df_prox['gex'])
        trocas_sinal = np.where(np.diff(df_prox['sign']) != 0)[0]
        
        if len(trocas_sinal) > 0:
            idx_flip = trocas_sinal[np.argmin(np.abs(df_prox.iloc[trocas_sinal]['strike'] - spot_convertido))]
            gamma_flip = df_prox.iloc[idx_flip]['strike']
        else:
            gamma_flip = df_prox.loc[df_prox['gex'].abs().idxmin()]['strike']
    else:
        gamma_flip = spot_convertido

    metricas = {
        'spot': spot_convertido,
        'call_wall': call_wall,
        'put_wall': put_wall,
        'key_level': key_level,
        'gamma_flip': gamma_flip
    }

    return df_gex, metricas, None


# ------------------------------------------------------------------
# DESENHO DO GRÁFICO
# ------------------------------------------------------------------
def plotar_grafico_gex(df_gex, metricas, titulo, e_pontos=False):
    fig = go.Figure()

    cores = ['#1b8a2e' if v >= 0 else '#ff3b30' for v in df_gex['gex']]
    fig.add_trace(go.Bar(
        y=df_gex['strike'],
        x=df_gex['gex'],
        orientation='h',
        marker_color=cores,
        name="Gamma Exposure"
    ))

    fmt = ".0f" if e_pontos else ".2f"

    fig.add_hline(y=metricas['spot'], line_dash="solid", line_color="yellow", annotation_text=f"Spot: {metricas['spot']:{fmt}}")
    fig.add_hline(y=metricas['call_wall'], line_dash="dash", line_color="#ab47bc", annotation_text=f"Call Wall: {metricas['call_wall']:{fmt}}")
    fig.add_hline(y=metricas['put_wall'], line_dash="dash", line_color="#ff3b30", annotation_text=f"Put Wall: {metricas['put_wall']:{fmt}}")
    fig.add_hline(y=metricas['gamma_flip'], line_dash="dash", line_color="#00e5ff", annotation_text=f"Gamma Flip: {metricas['gamma_flip']:{fmt}}")

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
# INTERFACE PRINCIPAL
# ------------------------------------------------------------------
st.title("PERFIL DE EXPOSIÇÃO DE GAMA (GEX)")

ativo = st.radio("Selecione o Mercado", ["EWZ (EUA / Brasil ETF)", "BOVA11 / WIN (B3)"], horizontal=True)

if ativo == "EWZ (EUA / Brasil ETF)":
    with st.spinner("Buscando dados de EWZ (Yahoo Finance)..."):
        df_gex, metricas, erro = obter_dados_gex_ewz(taxa_juros=0.045)
    
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
        st.error(f"Erro EWZ: {erro}")

else:
    with st.spinner("Buscando matriz de opções do BOVA11 diretamente do Opções.net..."):
        df_gex, metricas, erro = obter_dados_gex_bova11(taxa_di=0.1075)
        
    if df_gex is not None:
        col_graf, col_card = st.columns([3, 1])
        with col_graf:
            st.plotly_chart(plotar_grafico_gex(df_gex, metricas, "BOVA11 / Projeção WIN", e_pontos=True), use_container_width=True)
        with col_card:
            st.markdown("### Níveis em Pontos (WIN)")
            st.metric("Spot Projetado", f"{metricas['spot']:.0f} pts")
            st.metric("Call Wall", f"{metricas['call_wall']:.0f} pts")
            st.metric("Put Wall", f"{metricas['put_wall']:.0f} pts")
            st.metric("Key Level", f"{metricas['key_level']:.0f} pts")
            st.metric("Gamma Flip", f"{metricas['gamma_flip']:.0f} pts")
    else:
        st.warning(f"Indisponível: {erro}")
