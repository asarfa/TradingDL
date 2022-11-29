from enum import Enum


class EnumQuoteType(Enum):
    NEW = 0
    CANCEL = 1
    MODIFY = 2


class EnumCcy(Enum):
    """
    Enum with currencies
    """

    GBP = 1
    AUD = 2
    USD = 3
    EUR = 4
    JPY = 5
    CHF = 6
    NZD = 7
    NOK = 8
    SEK = 9
    CAD = 10


class EnumPair(Enum):
    """
    Enum with currency pairs
    """

    GBPUSD = 1
    EURUSD = 2
    EURJPY = 3
    GBPAUD = 4
    USDCHF = 5
    USDJPY = 6
    NZDUSD = 7
    AUDUSD = 8
    NOKSEK = 9
    USDCAD = 10
    OTHER = 99

    def get_ccy_first(self) -> EnumCcy:
        return EnumCcy[self.name[0:3]]

    def get_ccy_second(self):
        return EnumCcy[self.name[3:6]]

    def get_ccy_pair_with_slash(self):
        return self.name[0:3] + '/' + self.name[3:6]


class EnumQuoteFileType(Enum):
    """
    Enumerates quotes file types
    """

    LEVEL = True
    TICK = False