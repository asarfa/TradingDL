import Quote
from CommonUtilities import CommonUtilities
from EnumClasses import EnumPair
import re


class QuotesReaderOneCCY:

    def __init__(self, file_name: str, currency_pair_arg: EnumPair, info: bool = True) -> None:
        """
        Constructor for single currency reader.
        @param file_name: file path. Absolute or relative.
        @param currency_pair_arg: ENUM pair as EnumPair object. Containing the information of CCY Pair that is being
        monitored.
        @param info: True if you want to get reading status (True by default).
        @param debug:  true if you want to get more reading status (False by default).
        """

        self.__file_name = file_name
        self.__reader = open(self.__file_name)
        self.__is_reader_closed = False
        self.currency_pair_enum = currency_pair_arg
        self.currency_pair_str = currency_pair_arg.get_ccy_pair_with_slash()
        self.__new_pattern = fr"N;[a-zA-Z0-9_-]+;{self.currency_pair_str};[0-9]+;[0-9]+;[0-9].+"
        self.__cancel_pattern = fr"C;[a-zA-Z0-9_-]+;{self.currency_pair_str};[0-9]+;[0-9].+"
        self.__modify_pattern = fr"M;[a-zA-Z0-9_-]+;{self.currency_pair_str};[0-9]+;[0-9]+;[0-9].+"
        self._info = info

        if CommonUtilities.DEBUG:
            # Force INFO when debugging.
            self._info = True
            self.__line_by_line_reader_counter = 0

        if self._info:
            self.__line_quote_by_quote_reader_counter = 0

        if self._info:
            print("Created reader for file {}. Reading {} ccy pair.".format(self.__file_name, self.currency_pair_str))

    def read_line(self) -> Quote:
        """
        Function used to read one line of the CSV for one currency pair and returns a quote.
        Or None if the line was empty and there was nothing else to read.
        """
        if self.__is_reader_closed:
            return None

        line = self.__reader.readline()
        quote = None
        # Read the lines while you haven't met next QUOTE with necessary CCYies
        while quote is None:
            if CommonUtilities.DEBUG:
                self.__line_by_line_reader_counter += 1
            if not line:
                # Problem reading/end of file -> exit
                # Close: resource leakage
                self.__reader.close()
                self.__is_reader_closed = True
                if self._info:
                    print("Done reading file {}: no lines remaining in the file.".format(self.__file_name))
                return None
            if self._is_has_currency(line):
                quote = self.deserialize_quote(line)
            else:
                line = self.__reader.readline()

        # Found another quote.

        if CommonUtilities.DEBUG:
            self.__line_quote_by_quote_reader_counter += 1
        return quote

    def read_to_end(self) -> list:
        """
        Function used to read line by line the CSV for one currency pair and returns a list of quotes
        """
        if self.__is_reader_closed:
            return None

        instances_of_ccy_pair = self.__count_instances_of_ccy_pair()
        quotes_list = [None] * instances_of_ccy_pair
        total_found = 0
        while True:
            quote = self.read_line()
            if quote is not None:
                quotes_list[total_found] = quote
                total_found += 1
                if CommonUtilities.DEBUG:
                    if total_found % 10000 == 0:
                        print("Read {}/{} ({}%) quotes from file."
                              .format(total_found, instances_of_ccy_pair,
                                      round((total_found / instances_of_ccy_pair) * 100.0, 2)))
            else:
                break

        # Close: resource leakage
        self.__reader.close()
        self.__is_reader_closed = True

        return quotes_list

    def close_reader(self) -> None:
        """
        Release reader resources
        @return:
        """
        self.__reader.close()
        self.__is_reader_closed = True

    def deserialize_quote(self, quote_line: str) -> Quote:
        """
        Function that determines which quotes it is by using the regex
        The regex allows to determine if it is a modified, cancelled or new quotation

        """
        if bool(re.match(self.__new_pattern, quote_line)):
            line_list = CommonUtilities.split_quote_line_to_list(quote_line)[1:-1]
            return Quote.NewQuote(*line_list)
        elif bool(re.match(self.__cancel_pattern, quote_line)):
            line_list = CommonUtilities.split_quote_line_to_list(quote_line)[1:]
            return Quote.CancelQuote(*line_list)
        elif bool(re.match(self.__modify_pattern, quote_line)):
            line_list = CommonUtilities.split_quote_line_to_list(quote_line)[1:]
            return Quote.ModifyQuote(*line_list)

    def _is_has_currency(self, quote_line: str) -> bool:
        """
        Checks the line for the CCY pair in the known position.
        """
        count_of_point_comma: int = 0
        count_of_read_chars: int = 0
        count_chars = len(quote_line)
        while count_of_read_chars < count_chars:
            if quote_line[count_of_read_chars] == ";":
                count_of_point_comma += 1
            if count_of_point_comma == 2:
                # Check that we will not overflow the count of characters in this line.
                if count_of_read_chars + 8 < count_chars:
                    if quote_line[count_of_read_chars + 1:count_of_read_chars + 8] == self.currency_pair_str:
                        # Correct ccy pair
                        return True
                    else:
                        # Other ccy pair
                        return False
            if count_of_point_comma > 2:
                return False
            count_of_read_chars += 1
        # we haven't found enough ';'
        return False

    def __count_instances_of_ccy_pair(self) -> int:
        """
        Counts the number of CCY pair occurrences in the file
        """
        total_found = 0
        reader = open(self.__file_name)
        line = reader.readline()

        while line:
            if self._is_has_currency(line):
                total_found += 1
            line = reader.readline()

        reader.close()
        return total_found