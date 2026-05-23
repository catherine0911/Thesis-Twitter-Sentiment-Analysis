import re
import emoji

def preprocess_sarcasm(text):
    if not isinstance(text, str): 
        return ""
    # Replace @username with a generic @user token
    text = re.sub(r"@[^\s]+", "@user", text)
    # Replace web links with 'http'
    text = re.sub(r"http\S+", "http", text)
    # Convert emoji to text
    text = emoji.demojize(text, delimiters=(" :", ": "))
    # Remove hashtag related to sarcasm
    text = re.sub(r"#sarcasm|#sarcastic|#irony|#not", "", text, flags=re.IGNORECASE)

    return " ".join(text.split())