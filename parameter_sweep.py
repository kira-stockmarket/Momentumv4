def run_sweep():
    sig_df = generate_signals()
    unique_dates = sig_df.index.unique().sort_values()
    
    # CORRECTED: True daily compounding of 6% annual
    daily_rf = (1.0 + RISK_FREE_RATE) ** (1 / 252) - 1.0

    # Adjusted grid: Lowering thresholds to find the real optimal zone
    prob_thresholds = [0.65, 0.70, 0.75, 0.80]
    stop_losses = [-0.10, -0.12, -0.15]
    hold_days = [40, 50, 60]
    trail_activations = [0.25, 0.35]
    
    combinations = list(itertools.product(prob_thresholds, stop_losses, hold_days, trail_activations))
    results = []

    print(f"Executing Fast-Sweep across {len(combinations)} parameter sets...")

    for (prob, stop, hold, trail) in combinations:
        nav, cash = INITIAL_CAPITAL, INITIAL_CAPITAL
        open_positions = []
        portfolio_history = []

        for i in range(len(unique_dates) - 1):
            current_date, next_date = unique_dates[i], unique_dates[i + 1]
            cash *= (1.0 + daily_rf)

            todays_data = sig_df.loc[current_date]
            if isinstance(todays_data, pd.Series): todays_data = todays_data.to_frame().T
            stock_map = todays_data.set_index("Ticker").to_dict(orient="index")

            surviving_positions = []
            for pos in open_positions:
                ticker = pos["Ticker"]
                pos["Days_Held"] += 1

                if ticker not in stock_map:
                    surviving_positions.append(pos)
                    continue

                row = stock_map[ticker]
                pos["Highest_High"] = max(pos.get("Highest_High", pos["Entry_Price"]), row["High"])
                max_gain = (pos["Highest_High"] - pos["Entry_Price"]) / pos["Entry_Price"]

                low_ret = (row["Low"] - pos["Entry_Price"]) / pos["Entry_Price"]
                open_ret = (row["Open"] - pos["Entry_Price"]) / pos["Entry_Price"]

                current_stop = stop
                if max_gain >= trail: current_stop = max(stop, max_gain - 0.10)

                exit_trade = False
                if low_ret <= current_stop:
                    exit_trade = True
                    raw_return = min(current_stop, open_ret)
                elif pos["Days_Held"] >= hold:
                    exit_trade = True
                    raw_return = (row["Close"] - pos["Entry_Price"]) / pos["Entry_Price"]

                if exit_trade:
                    cash += pos["Allocated"] * (1.0 + (raw_return - ROUNDTRIP_FRICTION))
                else:
                    pos["Current_Value"] = pos["Allocated"] * (1.0 + (row["Close"] - pos["Entry_Price"]) / pos["Entry_Price"])
                    surviving_positions.append(pos)

            open_positions = surviving_positions
            nav = cash + sum(p["Current_Value"] for p in open_positions)
            portfolio_history.append(nav)

            eligible = todays_data[todays_data["Signal_Prob"] >= prob].sort_values(by="Signal_Prob", ascending=False)
            open_tickers = {p["Ticker"] for p in open_positions}
            slots = MAX_POSITIONS - len(open_positions)

            if slots > 0 and not eligible.empty:
                cands = eligible[~eligible["Ticker"].isin(open_tickers)].head(slots)
                next_day_data = sig_df.loc[next_date]
                if isinstance(next_day_data, pd.Series): next_day_data = next_day_data.to_frame().T
                next_open_map = next_day_data.set_index("Ticker")["Open"].to_dict()
                alloc = nav / MAX_POSITIONS

                for _, cand in cands.iterrows():
                    tk = cand["Ticker"]
                    if tk in next_open_map and cash >= alloc:
                        fill = next_open_map[tk]
                        if fill > 0:
                            cash -= alloc
                            open_positions.append({
                                "Ticker": tk, "Entry_Price": fill, "Allocated": alloc, 
                                "Current_Value": alloc, "Days_Held": 0, "Highest_High": fill
                            })

        if not portfolio_history: continue
        
        perf = pd.Series(portfolio_history)
        total_days = (unique_dates[-1] - unique_dates[0]).days
        cagr = ((perf.iloc[-1] / perf.iloc[0]) ** (365.25 / total_days)) - 1.0
        max_dd = ((perf - perf.cummax()) / perf.cummax()).min()
        
        daily_returns = perf.pct_change().dropna()
        sharpe = ((daily_returns.mean() - (RISK_FREE_RATE / 252)) / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        results.append({
            "Prob": prob, "Stop": stop, "Hold": hold, "Trail": trail, 
            "CAGR": cagr, "Sharpe": sharpe, "Max_DD": max_dd
        })

    res_df = pd.DataFrame(results).sort_values(by="Sharpe", ascending=False).head(15)
    print("\n--- TOP 15 REGIME-ADJUSTED PARAMETERS ---")
    print(res_df.to_string(index=False))
