from unittest import TestCase

from EnumClasses import EnumPair
from QuotesReaderOneCCY import QuotesReaderOneCCY
from Quote import Quote


class TestOneCcyLineReader(TestCase):

    def test_read_to_end_tick_file(self):
        """
        Function that test to reads the CSV lines for one currency
        """
        file_name: str = "data.part05/live-fm-log-11May-07-20-17-022.csv"
        currency_pair: EnumPair = EnumPair.EURUSD
        liste_quote: list = QuotesReaderOneCCY(file_name, currency_pair).read_to_end()
        self.assertEqual(7690, len(liste_quote))

    def test_read_by_line_tick_file(self):
        """
        Function that test to reads the CSV lines for one currency line by line
        """
        file_name: str = "test_tick_format.csv"
        currency_pair: EnumPair = EnumPair.EURUSD

        reader = QuotesReaderOneCCY(file_name, currency_pair)

        list_of_quotes = []
        next_quote: Quote = reader.read_line()

        while next_quote is not None:
            list_of_quotes.append(next_quote)
            next_quote = reader.read_line()

        self.assertEqual(7690, len(list_of_quotes))

    def test_read_to_end_level_file(self):
        """
        Function that test to reads the CSV lines for one currency
        """
        file_name: str = "test_level_format.csv"
        currency_pair: EnumPair = EnumPair.EURUSD
        liste_quote: list = QuotesReaderOneCCY(file_name, currency_pair).read_to_end()
        # Count how many EURUSD instances there is
        self.assertEqual(4496, len(liste_quote))

    def test_read_to_end_level_file_debug(self):
        """
        Function that test to reads the CSV lines for one currency. DEBUG ON.
        """
        file_name: str = "test_level_format.csv"
        currency_pair: EnumPair = EnumPair.EURUSD
        liste_quote: list = QuotesReaderOneCCY(file_name, currency_pair, True).read_to_end()
        # Count how many EURUSD instances there is
        self.assertEqual(4496, len(liste_quote))

    def test_read_by_line_level_file(self):
        """
        Function that test to reads the CSV lines for one currency line by line
        """
        file_name: str = "test_level_format.csv"
        currency_pair: EnumPair = EnumPair.EURUSD

        reader = QuotesReaderOneCCY(file_name, currency_pair)

        list_of_quotes = []
        next_quote: Quote = reader.read_line()

        while next_quote is not None:
            list_of_quotes.append(next_quote)
            next_quote = reader.read_line()

        self.assertEqual(4496, len(list_of_quotes))

    def test_read_by_line_level_file_debug(self):
        """
        Function that test to reads the CSV lines for one currency line by line. DEBUG ON.
        """
        file_name: str = "test_level_format.csv"
        currency_pair: EnumPair = EnumPair.EURUSD

        reader = QuotesReaderOneCCY(file_name, currency_pair, True)

        list_of_quotes = []
        next_quote: Quote = reader.read_line()

        while next_quote is not None:
            list_of_quotes.append(next_quote)
            next_quote = reader.read_line()

        self.assertEqual(4496, len(list_of_quotes))

    def test_read_file_with_problem(self):
        """
        Function that tries to read a file which has an unexpected ending.
        """
        file_name: str = "test_problem_file.csv"
        currency_pair: EnumPair = EnumPair.EURUSD

        reader = QuotesReaderOneCCY(file_name, currency_pair)

        list_of_quotes = []
        next_quote: Quote = reader.read_line()

        while next_quote is not None:
            list_of_quotes.append(next_quote)
            next_quote = reader.read_line()

        self.assertEqual(3692, len(list_of_quotes))