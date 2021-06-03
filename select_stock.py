# !/bin/python
import akshare as ak

# 实时行情
stock_zh_a_spot_df = ak.stock_zh_a_spot() 
print(stock_zh_a_spot_df)
# 历史行情(前复权)
stock_zh_a_daily_qfq_df = ak.stock_zh_a_daily(symbol="sz000002", start_date="20101103", end_date="20201116", adjust="qfq")
print(stock_zh_a_daily_qfq_df)
# 历史行情(后复权)
stock_zh_a_daily_hfq_df = ak.stock_zh_a_daily(symbol="sz000002", start_date='20201103', end_date='20201116', adjust="hfq")
print(stock_zh_a_daily_hfq_df)

if __name__=='__main__':
    print('run')