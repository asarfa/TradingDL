from EnumClasses import EnumCcy, EnumQuoteType, EnumPair


class Quote:
    # Static field with incrementing ID for the current quote.
    # This variable will be incremented after each call of TradeSituation.generate_next_id().
    # It is used to populate __quote_id.
    __common_quote_id: int = 0

    def __init__(self, quote_type: EnumQuoteType, quote_id_arg, currency1_arg: EnumCcy, currency2_arg: EnumCcy,
                 local_timestamp_arg: int, ecn_timestamp_arg: int, amount_arg: float = None, minqty_arg: float = None,
                 lotsize_arg: float = None, price_arg: float = None, way_arg: bool = False):
        """
        Initializes the instance of the Quote.
        """
        self.__quote_internal_id = Quote.generate_next_id()
        self.__quote_type = quote_type
        self.__quote_ecn_id = quote_id_arg
        self.__ccy1 = currency1_arg
        self.__ccy2 = currency2_arg
        self.__ccy_pair = EnumPair[EnumCcy(currency1_arg).name + EnumCcy(currency2_arg).name]
        self.__local_timestamp = local_timestamp_arg
        self.__ecn_timestamp = ecn_timestamp_arg
        self.__amount = amount_arg
        self.__minimum_quantity = minqty_arg
        self.__lot_size = lotsize_arg
        self.__price = price_arg
        self.__order_way: bool = way_arg

    def get_quote_type(self) -> EnumQuoteType:
        """
        Returns the quote type used to speed up the identification process for the order book
        @return: Quote type enum
        """
        return self.__quote_type

    def get_id_internal(self) -> int:
        """
        Returns the ID of this specific quote
        """
        return self.__quote_internal_id

    def get_id_ecn(self) -> str:
        return self.__quote_ecn_id

    def get_ccy1(self) -> EnumCcy:
        return self.__ccy1

    def get_ccy2(self) -> EnumCcy:
        return self.__ccy2

    def get_pair(self) -> EnumPair:
        """
        Returns the currency pair
        """
        return self.__ccy_pair

    def get_local_timestamp(self) -> int:
        # Usually counted in nano seconds
        if self.__local_timestamp is None:
            return None
        else:
            return int(self.__local_timestamp)

    def get_ecn_timestamp(self) -> int:
        if self.__ecn_timestamp is None:
            return None
        else:
            return int(self.__ecn_timestamp)

    def get_amount(self) -> float:
        if self.__amount is None:
            return None
        else:
            return float(self.__amount)

    def get_min_quant(self) -> float:
        if self.__minimum_quantity is None:
            return 0.00
        else:
            return float(self.__minimum_quantity)

    def get_lot_size(self) -> float:
        if self.__lot_size is None:
            return 0.00
        else:
            return float(self.__lot_size)

    def get_price(self) -> float:
        if self.__price is None:
            return None
        else:
            return float(self.__price)

    def get_way(self) -> bool:
        return self.__order_way

    @staticmethod
    def generate_next_id():
        Quote.__common_quote_id += 1
        return Quote.__common_quote_id


class NewQuote(Quote):

    def __init__(self, quote_id_arg, currency1_arg, currency2_arg, local_timestamp_arg, ecn_timestamp_arg, amount_arg,
                 minqty_arg, lotsize_arg, price_arg, way_arg):
        """
         it's lets the class new quote initialize the object's attributes
         The class heritates from the class Quotes
         """
        currency1_enum = EnumCcy[currency1_arg].value
        currency2_enum = EnumCcy[currency2_arg].value
        way_bool = way_arg == 'B'
        # try converting to int
        if quote_id_arg.isdigit():
            quote_id = int(quote_id_arg)
        else:
            quote_id = quote_id_arg
        Quote.__init__(self, EnumQuoteType.NEW, quote_id, currency1_enum, currency2_enum, local_timestamp_arg,
                       ecn_timestamp_arg, amount_arg, minqty_arg, lotsize_arg, price_arg, way_bool)


class CancelQuote(Quote):

    def __init__(self, quote_id_arg, currency1_arg, currency2_arg, local_timestamp_arg, ecn_timestamp_arg):
        """
         it's lets the class cancelled quote initialize the object's attributes
         The class heritates from the class Quotes
        """
        currency1_enum = EnumCcy[currency1_arg].value
        currency2_enum = EnumCcy[currency2_arg].value
        # try converting to int
        if quote_id_arg.isdigit():
            quote_id = int(quote_id_arg)
        else:
            quote_id = quote_id_arg
        Quote.__init__(self, EnumQuoteType.CANCEL, quote_id, currency1_enum, currency2_enum, local_timestamp_arg,
                       ecn_timestamp_arg)


class ModifyQuote(Quote):

    def __init__(self, quote_id_arg, currency1_arg, currency2_arg, local_timestamp_arg, ecn_timestamp_arg, amount_arg,
                 minqty_arg, lotsize_arg, price_arg):
        """
         it's lets the class modified quote initialize the object's attributes
         The class heritates from the class Quotes
         """
        currency1_enum = EnumCcy[currency1_arg].value
        currency2_enum = EnumCcy[currency2_arg].value
        # try converting to int
        if quote_id_arg.isdigit():
            quote_id = int(quote_id_arg)
        else:
            quote_id = quote_id_arg
        Quote.__init__(self, EnumQuoteType.MODIFY, quote_id, currency1_enum, currency2_enum, local_timestamp_arg,
                       ecn_timestamp_arg, amount_arg, minqty_arg, lotsize_arg, price_arg)


@staticmethod
def create_new_from_modify(modify_quote: Quote, original_way: bool) -> Quote:
    """
    Creates a New quote from a Modify quote.
    @param modify_quote: incoming MODIFY quote
    @param original_way: the B or S characteristic from the original order. Mandatory.
    @return: New quote
    """
    new_quote = Quote(EnumQuoteType.NEW, modify_quote.get_id_ecn(), modify_quote.get_ccy1(), modify_quote.get_ccy2(),
                      modify_quote.get_local_timestamp(), modify_quote.get_ecn_timestamp(),
                      modify_quote.get_amount(), modify_quote.get_min_quant(), modify_quote.get_lot_size(),
                      modify_quote.get_price(), original_way)
    return new_quote