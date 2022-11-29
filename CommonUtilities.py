import os
import glob
import re
from itertools import (takewhile, repeat)
from EnumClasses import EnumQuoteFileType, EnumQuoteType
from Quote import Quote
import hashlib


def asc_key_fn(key):
    """
    This is a key sort function. It's used in the SortedDict collection
    @param key: provided by the SortedDict
    @return:
    """
    return key


def dsc_key_fn(key):
    """
    This is a key sort function
    @param key: provided by the SortedDict
    @return:
    """
    return -key


class CommonUtilities:
    """
    Utility class. No instance methods and constructors here
    """
    DEBUG = True
    # Determines how many digits to keep in the fraction.
    FLOAT_ROUND_PRECISION: int = 8
    # How many digits there is after the comma in the fractions
    PRICE_ROUND_PRECISION: int = 5
    # Nanos in second, millisecond, minute etc.
    NANOS_IN_ONE_MILLIS = 1000000
    NANOS_IN_ONE_SECOND = 1000000000
    NANOS_IN_ONE_MINUTE = 60 * NANOS_IN_ONE_SECOND
    MILLIS_IN_ONE_SECOND = 1000
    MILLIS_IN_ONE_MINUTE = 60 * MILLIS_IN_ONE_SECOND

    @staticmethod
    def num_of_rows(address: str) -> int:
        """
        Function that counts the number of rows in the data file
        Buffer read strategy. rawincount method as in
        https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
        @param address: the path to the FILE with quotes
        @return: count of rows in that file
        """
        f = open(address, 'rb')
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        # Add +1 because on the last line we could miss the CR LF char
        count = sum(buf.count(b'\n') for buf in bufgen) + 1
        f.close()
        return count

    @staticmethod
    def get_csv_list(address: str) -> list:
        """
        Function that returns a list of all the csv that exists in the directory
        """
        os.chdir(address)
        csv_list = glob.glob('*.csv')
        return csv_list

    @staticmethod
    def get_pkl_list(address: str) -> list:
        """
        Function that returns a list of all the csv that exists in the directory
        """
        os.chdir(address)
        pkl_list = glob.glob('*.pkl')
        return pkl_list

    @staticmethod
    def split_quote_line_to_list(quote_line: str) -> list:
        """
        Function that split string in the line by the occurrences of pattern
        Transform the line to a list
        """
        quote_line = quote_line.rstrip('\r\n ')
        line_list = re.split(';', quote_line)
        ccies = line_list[2][0:3], line_list[2][4:8]
        del line_list[2]
        line_list[2:2] = ccies
        return line_list

    @staticmethod
    def precision_round(number: float) -> float:
        """
        Truncates the number to a managable/comparable size. Helps preventing the situation where you might have too
        many numbers in the fraction.
        @param number: truncated number
        @return: truncated number
        """
        return round(number, CommonUtilities.FLOAT_ROUND_PRECISION)

    @staticmethod
    def compare_numbers(number1: float, number2: float) -> bool:
        """
        Compares two numbers using the precision rounding defined in CommonUtilities.precision_round
        @param number1:
        @param number2:
        @return: true if numbers are equal
        """
        if CommonUtilities.precision_round(number1) == CommonUtilities.precision_round(number2):
            return True
        return False

    @staticmethod
    def return_file_type(file_name: str) -> EnumQuoteFileType:
        reader = open(file_name)
        quote_line = reader.readline()
        reader.close()
        if "-" in quote_line:
            return EnumQuoteFileType.LEVEL
        else:
            return EnumQuoteFileType.TICK

    @staticmethod
    def create_new_from_modify(modify_quote: Quote, original_way: bool) -> Quote:
        """
        Creates a New quote from a Modify quote.
        @param modify_quote: incoming MODIFY quote
        @param original_way: the B or S characteristic from the original order. Mandatory.
        @return: New quote
        """
        new_quote = Quote(EnumQuoteType.NEW, modify_quote.get_id_ecn(), modify_quote.get_ccy1(),
                          modify_quote.get_ccy2(),
                          modify_quote.get_local_timestamp(), modify_quote.get_ecn_timestamp(),
                          modify_quote.get_amount(), modify_quote.get_min_quant(), modify_quote.get_lot_size(),
                          modify_quote.get_price(), original_way)
        return new_quote

    @staticmethod
    def return_md5_string_hash(string_to_hash: str, **kwargs) -> int:
        """
        Returns a hash of string given in input (as integer)
        :param string_to_hash: text that will be hashed
        :param length: maximal length of the returned integer
        :return: int representing the provided string
        """
        if 'length' in kwargs and kwargs['length'] is not None and kwargs['length'] >= 10:
            # for some reason the length is + 2
            return int(hashlib.md5(string_to_hash.encode('utf-8')).hexdigest()[:kwargs['length'] - 1], 16)
        else:
            return int(hashlib.md5(string_to_hash.encode('utf-8')).hexdigest()[:15], 16)