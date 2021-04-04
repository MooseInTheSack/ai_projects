import requests

f = open("../apiKey.txt", "r")
apiKey = f.read()

def getAlphaVantageInfo(companySymbol):
    from alpha_vantage.timeseries import TimeSeries
    ts = TimeSeries(key=apiKey)
    # Get json object with the intraday data and another with  the call's metadata
    data, meta_data = ts.get_intraday(symbol=companySymbol, outputsize='full')

    print(data)

def getCompanyInfo(function,companySymbol, apiKey):
    #function = "OVERVIEW", "EARNINGS", "INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", "EARNINGS_CALENDAR"
    payload = { 'function': function, 'symbol': companySymbol, 'apikey': apiKey, }
    response = requests.get("https://www.alphavantage.co/query", params=payload)
    return response.json()

#getAlphaVantageInfo('MSFT')
#getCompanyInfo('OVERVIEW', 'IBM', apiKey)
googBalanceSheet = getCompanyInfo('BALANCE_SHEET', 'GOOGL', apiKey)
googIncomeStatement = getCompanyInfo('INCOME_STATEMENT', 'GOOGL', apiKey)

print(googIncomeStatement)