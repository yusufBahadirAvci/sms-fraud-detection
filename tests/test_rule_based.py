"""Tests for the rule-based fraud checker.

pytest finds every function whose name starts with "test_" and runs it.
"assert" checks that something is True. If it is False, the test fails.
Tests protect us: if we change the code later and break something,
a test will tell us right away.
"""

from src.sms_fraud.rule_based import is_fraud


def test_fraud_message_is_detected():
    # This message has the keyword "kazandınız", so it must be fraud.
    assert is_fraud("Tebrikler, 5000 TL kazandınız!") is True


def test_normal_message_is_not_fraud():
    # A normal daily message has no suspicious words.
    assert is_fraud("Yarın sabah 9'da ofiste görüşelim.") is False


def test_uppercase_input_is_detected():
    # Fraud SMS often use CAPITAL letters to look urgent.
    # Our check must work no matter if letters are big or small.
    # We test English and Turkish, because Turkish "I/İ" is a special case.
    assert is_fraud("WINNER! CLICK HERE NOW") is True
    assert is_fraud("TEBRİKLER, KAZANDINIZ!") is True


def test_empty_string_is_not_fraud():
    # An empty message must not crash the program and must not be fraud.
    assert is_fraud("") is False
