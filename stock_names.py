sp500 = ['AOS', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADM', 'ADP', 'AAP', 'AIG', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE',
                   'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AMD', 'AEE', 'AAL', 'AEP',
                   'AXP', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'ANTM', 'AON', 'APA', 'AAPL',
                   'AMAT', 'APTV', 'ANET', 'AIZ', 'T', 'ATO', 'ADSK', 'AZO', 'AVB', 'AVY', 'BKR', 'BLL', 'BAC', 'BBWI',
                   'BAX', 'BDX', 'WRB', 'BBY', 'BIO', 'TECH', 'BIIB', 'BLK', 'BK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX',
                   'BMY', 'AVGO', 'BR', 'BRO', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR',
                   'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CDAY', 'CERN', 'CF', 'CRL', 'SCHW',
                   'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME',
                   'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW',
                   'CTVA', 'COST', 'CTRA', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY',
                   'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISH', 'DIS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE',
                   'DUK', 'DRE', 'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ENPH', 'ETR',
                   'EOG', 'EPAM', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'RE', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD',
                   'EXR', 'XOM', 'FFIV', 'FDS', 'FAST', 'FRT', 'FDX', 'FITB', 'FRC', 'FE', 'FIS', 'FISV', 'FLT', 'FMC',
                   'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX', 'AJG', 'GRMN', 'IT', 'GE', 'GNRC', 'GD',
                   'GIS', 'GPC', 'GILD', 'GL', 'GPN', 'GM', 'GS', 'GWW', 'HAL', 'HIG', 'HAS', 'HCA', 'PEAK', 'HSIC',
                   'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUM', 'HII', 'HBAN',
                   'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG',
                   'IVZ', 'IPGP', 'IQV', 'IRM', 'JBHT', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KEY', 'KEYS',
                   'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LLY',
                   'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LUMN', 'LYB', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR',
                   'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'FB', 'MET', 'MTD', 'MGM',
                   'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS',
                   'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI',
                   'NDSN', 'NSC', 'NTRS', 'NOC', 'NLOK', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY',
                   'ODFL', 'OMC', 'OKE', 'ORCL', 'OGN', 'OTIS', 'PCAR', 'PKG', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL',
                   'PENN', 'PNR', 'PEP', 'PKI', 'PFE', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG',
                   'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RL',
                   'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL',
                   'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'NOW', 'SHW', 'SBNY', 'SPG', 'SWKS', 'SJM', 'SNA',
                   'SEDG', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STE', 'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'TMUS',
                   'TROW', 'TTWO', 'TPR', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO',
                   'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TWTR', 'TYL', 'TSN', 'USB', 'UDR', 'ULTA', 'UAA', 'UA', 'UNP',
                   'UAL', 'UNH', 'UPS', 'URI', 'UHS', 'VLO', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC', 'VTRS', 'V',
                   'VNO', 'VMC', 'WAB', 'WMT', 'WBA', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WRK',
                   'WY', 'WHR', 'WMB', 'WTW', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']

russian_stocks = [
    "GAZP.ME",
    "ROSN.ME",
    "OMZZP.ME",
    "GMKN.ME",
    "NVTK.ME",
    "LKOH.ME",
    "SBER.ME",
    "SBERP.ME",
    "SIBN.ME",
    "PLZL.ME",
    "SNGSP.ME",
    "RUAL.ME",
    "NLMK.ME",
    "PHOR.ME",
    "CHMF.ME",
    "TATN.ME",
    "SNGS.ME",
    "TRNFP.ME",
    "TATNP.ME",
    "VSMO.ME",
    "AKRN.ME",
    "ALRS.ME",
    "YNDX.ME",
    "MAGN.ME",
    "MGNT.ME",
    "TCSG.ME",
    "PIKK.ME",
    "MTSS.ME",
    "ENPG.ME",
    "UNAC.ME",
    "HYDR.ME",
    "POLY.ME",
    "FIVE.ME",
    "RASP.ME",
    "VTBR.ME",
    "MOEX.ME",
    "RTKMP.ME",
    "IRAO.ME",
    "RTKM.ME",
    "BANEP.ME",
    "CBOM.ME",
    "BANE.ME",
    "NKNCP.ME",
    "MGTS.ME",
    "NKNC.ME",
    "MGTSP.ME",
    "KZOSP.ME",
    "RSTIP.ME",
    "RSTI.ME",
    "KZOS.ME",
    "FEES.ME",
    "AFKS.ME",
    "AGRO.ME",
    "GCHE.ME",
    "NMTP.ME",
    "FESH.ME",
    "YAKG.ME",
    "FLOT.ME",
    "APTK.ME",
    "LSNG.ME",
    "UPRO.ME",
    "IRKT.ME",
    "LSNGP.ME",
    "KAZT.ME",
    "KMAZ.ME",
    "AVAN.ME",
    "AFLT.ME",
    "KAZTP.ME",
    "MTLR.ME",
    "MSNG.ME",
    "MTLRP.ME",
    "RGSS.ME",
    "TRMK.ME",
    "DSKY.ME",
    "MFGSP.ME",
    "INGR.ME",
    "LSRG.ME",
    "OGKB.ME",
    "MSRS.ME",
    "UKUZ.ME",
    "SELG.ME",
    "MVID.ME",
    "UTAR.ME",
    "AQUA.ME",
    "BELU.ME",
    "MFGS.ME",
    "SPBE.ME",
    "MRKS.ME",
    "TGKA.ME",
    "RNFT.ME",
    "BSPB.ME",
    "VJGZ.ME",
    "USBN.ME",
    "TGKDP.ME",
    "TGKD.ME",
    "ROLO.ME",
    "QIWI.ME",
    "ETLN.ME",
    "MRKK.ME",
    "VJGZP.ME",
    "MSTT.ME",
    "SFIN.ME",
    "MRKP.ME",
    "NKHP.ME",
    "KUBE.ME",
    "DVEC.ME",
    "ENRU.ME",
    "ABRD.ME",
    "JNOSP.ME",
    "JNOS.ME",
    "UCSS.ME",
    "MRKU.ME",
    "LNZL.ME",
    "LNZLP.ME",
    "TTLK.ME",
    "BLNG.ME",
    "RKKE.ME",
    "MRKC.ME",
    "CHMK.ME",
    "AMEZ.ME",
    "KOGK.ME",
    "VRSB.ME",
    "PRMB.ME",
    "WTCMP.ME",
    "TNSE.ME",
    "WTCM.ME",
    "PAZA.ME",
    "KCHEP.ME",
    "KCHE.ME",
    "GAZA.ME",
    "PMSBP.ME",
    "ZILL.ME",
    "PMSB.ME",
    "NNSBP.ME",
    "NNSB.ME",
    "URKZ.ME",
    "VSYD.ME",
    "VSYDP.ME",
    "KRSBP.ME",
    "BRZL.ME",
    "KRSB.ME",
    "KRKN.ME",
    "RUSI.ME",
    "GAZAP.ME",
    "MRKV.ME",
    "KRKNP.ME",
    "SVAV.ME",
    "MRKY.ME",
    "KGKCP.ME",
    "KRKOP.ME",
    "KGKC.ME",
    "ISKJ.ME",
    "SAGO.ME",
    "RTSBP.ME",
    "RTSB.ME",
    "RZSB.ME",
    "VRSBP.ME",
    "UNKL.ME",
    "MGNZ.ME",
    "TGKBP.ME",
    "TGKB.ME",
    "TGKN.ME",
    "BISVP.ME",
    "YRSBP.ME",
    "CHKZ.ME",
    "YRSB.ME",
    "KMEZ.ME",
    "MAGE.ME",
    "RTGZ.ME",
    "IGSTP.ME",
    "EELT.ME",
    "MAGEP.ME",
    "KROT.ME",
    "MRKZ.ME",
    "IGST.ME",
    "RDRB.ME",
    "YKEN.ME",
    "KBSB.ME",
    "YKENP.ME",
    "KROTP.ME",
    "NFAZ.ME",
    "HIMCP.ME",
    "NAUK.ME",
    "MISB.ME",
    "CNTLP.ME",
    "MISBP.ME",
    "CNTL.ME",
    "ZVEZ.ME",
    "GTRK.ME",
    "NKSH.ME",
    "LPSB.ME",
    "VGSB.ME",
    "VGSBP.ME",
    "LIFE.ME",
    "TORS.ME",
    "TORSP.ME",
    "GEMA.ME",
    "CHGZ.ME",
    "RBCM.ME",
    "SLEN.ME",
    "RUGR.ME",
    "DZRDP.ME",
    "DZRD.ME",
    "TUZA.ME",
    "TASBP.ME",
    "ROST.ME",
    "SAREP.ME",
    "STSBP.ME",
    "TASB.ME",
    "ARSA.ME",
    "KLSB.ME",
    "STSB.ME",
    "LVHK.ME",
    "SARE.ME",
    "DIOD.ME",
    "KTSBP.ME",
    "KTSB.ME",
    "ASSB.ME",
    "MRSB.ME",
    "KUZB.ME",
    "VLHZ.ME",
    "NSVZ.ME",
    "ELTZ.ME",
    "MDMG.ME",
    "RENI.ME"
]

ishares_etf_list = [
    "IVV",
    "IEFA",
    "AGG",
    "IWF",
    "IJH",
    "IJR",
    "IEMG",
    "IWM",
    "IWD",
    "ITOT",
    "EFA",
    "TLT",
    "IVW",
    "QUAL",
    "MUB",
    "IXUS",
    "IWB",
    "IWR",
    "IVE",
    "LQD",
    "MBB",
    "IAU",
    "IEF",
    "IUSB",
    "DGRO",
    "GOVT",
    "SHY",
    "USMV",
    "IGSB",
    "SGOV",
    "ACWI",
    "TIP",
    "DVY",
    "EEM",
    "SHV",
    "IBIT",
    "EWJ",
    "IUSG",
    "EFV",
    "IUSV",
    "IYW",
    "HYG",
    "IWP",
    "PFF",
    "IWS",
    "EMB",
    "IWV",
    "IDEV",
    "IEI",
    "EMXC",
    "IGIB",
    "ESGU",
    "SOXX",
    "IWN",
    "EFG",
    "SLV",
    "OEF",
    "USHY",
    "IWO",
    "USIG",
    "HDV",
    "MTUM",
    "SCZ",
    "IJK",
    "INDA",
    "IWY",
    "SUB",
    "IQLT",
    "ESGD",
    "STIP",
    "TLH",
    "EZU",
    "IJJ",
    "IGV",
    "IBB",
    "DYNF",
    "FLOT",
    "VLUE",
    "IJS",
    "EFAV",
    "TFLO",
    "HEFA",
    "ITA",
    "SHYG",
    "IAGG",
    "IJT",
    "ICSH",
    "IHI",
    "EWY",
    "IOO",
    "EWZ",
    "MCHI",
    "ACWX",
    "FXI",
    "IXN",
    "IEUR",
    "EEMV",
    "DSI",
    "IGM",
    "ACWV",
    "IYR",
    "IDV",
    "ESGE",
    "EWT",
    "ISTB",
    "IXJ",
    "EAGG",
    "IGF",
    "REET",
    "IXC",
    "SUSA",
    "IYH",
    "XT",
    "URTH",
    "GVI",
    "ITB",
    "NEAR",
    "IBTE",
    "EWC",
    "CMF",
    "BINC",
    "IBDP",
    "EWU",
    "IBDQ",
    "IYF",
    "IBDR",
    "ICLN",
    "IFRA",
    "AAXJ",
    "USRT",
    "IBDS",
    "IBTF",
    "IGLB",
    "EWW",
    "IWX",
    "SLQD",
    "ILCG",
    "IPAC",
    "IMTM",
    "IMCG",
    "ICF",
    "AOR",
    "USCL",
    "EPP",
    "IYY",
    "EWA",
    "LRGF",
    "AOA",
    "ICVT",
    "IVLU",
    "IBDT",
    "PABU",
    "FALN",
    "ILF",
    "IEV",
    "ESML",
    "IYJ",
    "AIA",
    "LCTU",
    "IYE",
    "EUFN",
    "AOM",
    "IYG",
    "IYK",
    "IWL",
    "IBDU",
    "IAUM",
    "SUSL",
    "PICK",
    "EWL",
    "GSG",
    "SUSC",
    "SMMD",
    "EWP",
    "QLTA",
    "INTF",
    "IYC",
    "SMLF",
    "EWG",
    "IBTG",
    "USXF",
    "IYT",
    "CRBN",
    "ILCV",
    "TLTW",
    "IWC",
    "ILCB",
    "IDU",
    "INDY",
    "IEO",
    "IMCB",
    "SMIN",
    "SUSB",
    "KXI",
    "IBDV",
    "KSA",
    "IHAK",
    "IHF",
    "IBDW",
    "SMMV",
    "IBTH",
    "EUSA",
    "IHE",
    "DVYE",
    "NYF",
    "COMT",
    "EWQ",
    "IRBO",
    "IAK",
    "IGRO",
    "DMXF",
    "AGZ",
    "MEAR",
    "IYM",
    "IAT",
    "IMCV",
    "EUSB",
    "REZ",
    "AOK",
    "REM",
    "ILTB",
    "IGEB",
    "IGE",
    "EMGF",
    "ECH",
    "EXI",
    "ISCF",
    "IBDX",
    "ISCG",
    "IBMN",
    "HYDB",
    "IGOV",
    "EWH",
    "EIDO",
    "IAI",
    "IBMM",
    "RING",
    "IBHD",
    "IBHE",
    "IBTI",
    "FM",
    "LEMB",
    "IBMO",
    "CMBS",
    "EWS",
    "CEMB",
    "EWI",
    "IBMP",
    "ISCV",
    "EEMA",
    "HEZU",
    "TECB",
    "IXG",
    "IBDY",
    "EMHY",
    "EEMS",
    "HYBB",
    "LCTD",
    "IBHF",
    "SIZE",
    "BGRN",
    "DIVB",
    "GNMA",
    "EPOL",
    "LQDH",
    "IBMQ",
    "HEWJ",
    "IEZ",
    "SDG",
    "EDEN",
    "EWD",
    "EWN",
    "IXP",
    "EZA",
    "EWJV",
    "RXI",
    "MXI",
    "EWM",
    "IDRV",
    "XVV",
    "THD",
    "CMDY",
    "IETC",
    "GBF",
    "IYZ",
    "HYGH",
    "TUR",
    "IBTJ",
    "ISCB",
    "IBTK",
    "HAWX",
    "EWZS",
    "TOK",
    "WOOD",
    "SLVP",
    "GOVZ",
    "IBTM",
    "CNYA",
    "ISVL",
    "XJH",
    "LQDW",
    "IMTB",
    "IBTL",
    "BYLD",
    "IBTO",
    "EIS",
    "HEEM",
    "IDNA",
    "VEGI",
    "GHYG",
    "SCJ",
    "HYXF",
    "HSCZ",
    "EPHE",
    "GLOF",
    "JXI",
    "EIRL",
    "IYLD",
    "CLOA",
    "IEUS",
    "IFGL",
    "EPU",
    "FILL",
    "IGBH",
    "IBHG",
    "IBMR",
    "HYMU",
    "ENZL",
    "JPXN",
    "ISHG",
    "BRTR",
    "QAT",
    "EMXF",
    "SVAL",
    "KWT",
    "BKF",
    "XJR",
    "IBHH",
    "FIBR",
    "HYGW",
    "EWO",
    "IEDI",
    "PABD",
    "ECNS",
    "HYXU",
    "STLG",
    "CCRV",
    "LQDI",
    "IVVB",
    "IDGT",
    "DVYA",
    "BEMB",
    "UAE",
    "EWUS",
    "IAUF",
    "IBHI",
    "BRLN",
    "IVVM",
    "WPS",
    "HEWG",
    "LQDB",
    "LDEM",
    "BALI",
    "ENOR",
    "EAOA",
    "EMIF",
    "EWK",
    "CALY",
    "INMU",
    "EAOR",
    "FOVL",
    "IBLC",
    "EFNL",
    "IVVW",
    "IBHJ",
    "IBIC",
    "IWMW",
    "BTEK",
    "IBIB",
    "IBID",
    "IBIE",
    "IBIJ",
    "TMET",
    "IBIA",
    "USBF",
    "ELQD",
    "ESMV",
    "IBIG",
    "IBIF",
    "IBII",
    "ITDC",
    "ERET",
    "IBAT",
    "IBIH",
    "EAOK",
    "ICOP",
    "ITDB",
    "BLCV",
    "IDAT",
    "BLCR",
    "BTHM",
    "ITDF",
    "IVRS",
    "TCHI",
    "ITDE",
    "ITDD",
    "IWTR",
    "EAOM",
    "EGUS",
    "INRO",
    "ISZE",
    "AGRH",
    "IVEG",
    "BPAY",
    "EFRA",
    "ITDG",
    "EVUS",
    "BMED",
    "IBRN",
    "ITDI",
    "ETEC",
    "IRTR",
    "BECO",
    "IWFH",
    "ITDH",
    "ILIT",
    "ITDA",
    "HYGI",
    "AGIH"
]
