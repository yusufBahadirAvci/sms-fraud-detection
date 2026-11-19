"""A simple rule-based SMS fraud checker (no machine learning).

Idea: if a message contains a "suspicious" word or phrase, we say it is fraud.

Why do we build this?
This is our BASELINE. Later we will train an ML model. Then we can ask:
"Is the ML model really better than these simple rules?"
Without a baseline, we cannot answer that question.
"""

# A list of suspicious words and phrases.
# We write them in lowercase, because we will also make the message
# lowercase before we compare. This makes the check case-insensitive.
# We use Turkish AND English, because users can get SMS in both languages.
SUSPICIOUS_KEYWORDS = [
    # Turkish
    "kazandınız",  # "you won" - fake prize messages
    "tıklayın",  # "click" - tries to make you open a bad link
    "hesabınız bloke",  # "your account is blocked" - creates fear
    "ödül",  # "prize"
    "acil",  # "urgent" - pushes you to act fast
    "şifrenizi",  # "your password" - tries to steal passwords
    # English
    "winner",
    "click here",
    "you have won",
    "free gift",
    "verify your account",
    "urgent",
]


def normalize(text: str) -> str:
    """Make text lowercase and fix the Turkish "i" problem.

    Why is this needed?
    In Turkish there are two kinds of "i": "ı/I" (no dot) and "i/İ" (with dot).
    Python's lower() follows English rules, so "KAZANDINIZ".lower() gives
    "kazandiniz" (with dotted i), NOT "kazandınız". Then our keyword would
    not match. To avoid this, we turn every "ı" into a normal "i".
    We do the same to the keywords, so both sides look the same.
    """
    text = text.lower()
    # "İ".lower() gives "i" + an invisible dot mark (U+0307). Remove the mark.
    text = text.replace("̇", "")
    # Treat dotless "ı" the same as normal "i".
    text = text.replace("ı", "i")
    return text


def is_fraud(message: str) -> bool:
    """Return True if the message looks like fraud, otherwise False.

    The rule is very simple: if ANY suspicious keyword is inside the
    message, we return True.
    """
    # An empty message has no words, so it cannot be fraud.
    # We check this first to keep the logic clear.
    if not message:
        return False

    clean_message = normalize(message)

    # Check each keyword one by one.
    for keyword in SUSPICIOUS_KEYWORDS:
        # "in" checks if the keyword appears anywhere in the message.
        if normalize(keyword) in clean_message:
            # We found one suspicious keyword. That is enough, so stop here.
            return True

    # We checked all keywords and found nothing suspicious.
    return False


# This block runs ONLY when we run this file directly, for example:
#     python -m src.sms_fraud.rule_based
# It does NOT run when another file (like a test) imports this module.
# This lets us use the file both as a small demo and as a library.
if __name__ == "__main__":
    # 6 example messages: the first 3 are fraud, the last 3 are normal.
    # Each item is a pair: (message, expected answer).
    samples = [
        ("Tebrikler! 10.000 TL kazandınız. Hemen linke tıklayın.", True),
        ("Hesabınız bloke edilmiştir. Açmak için bilgilerinizi girin.", True),
        ("WINNER! Click here to claim your free iPhone now!", True),
        ("Akşam yemeğe geliyor musun?", False),
        ("Your package will arrive tomorrow between 9 and 12.", False),
        ("Toplantı saat 14:00'te başlayacak.", False),
    ]

    print("Rule-based SMS fraud check")
    print("=" * 60)

    correct = 0
    for message, expected in samples:
        result = is_fraud(message)
        label = "FRAUD " if result else "NORMAL"
        # We also show if the rule gave the answer we expected.
        # This is the first, very small "evaluation" of our baseline.
        status = "OK" if result == expected else "WRONG"
        if result == expected:
            correct += 1
        print(f"[{label}] ({status})  {message}")

    print("=" * 60)
    print(f"Correct: {correct} / {len(samples)}")
