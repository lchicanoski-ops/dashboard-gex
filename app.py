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

    # Níveis Chave
    call_wall = df_gex.loc[df_gex['gex'].idxmax()]['strike']
    put_wall = df_gex.loc[df_gex['gex'].idxmin()]['strike']
    key_level = df_gex.loc[df_gex['gex'].abs().idxmax()]['strike']
    
    # Cálculo Preciso do Gamma Flip: Ponto onde o saldo acumulado cruza o zero próximo ao Spot
    df_gex = df_gex.sort_values('strike').reset_index(drop=True)
    df_gex['cumsum'] = df_gex['gex'].cumsum()
    
    # Encontra a mudança de sinal na coluna cumsum
    zero_crossings = np.where(np.diff(np.sign(df_gex['cumsum'])))[0]
    
    if len(zero_crossings) > 0:
        # Pega o cruzamento mais próximo do preço atual (Spot)
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
