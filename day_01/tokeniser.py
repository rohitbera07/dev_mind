import tiktoken

class Tokeniser:

    def __init__(self, model_name="cl100k_base"):
        self.model_name = model_name
        self.encoding = tiktoken.get_encoding(model_name)
        self.token_ids = []

    def inspect(self, text="what is your name? mine is suryavanshi"):
        self.token_ids = self.encoding.encode(text)
        print("Token IDs:", self.token_ids)

    def decode(self):
        tokens = []

        for token_id in self.token_ids:
            token = self.encoding.decode_single_token_bytes(token_id)
            tokens.append(token.decode("utf-8", errors="replace"))

        print("Decoded tokens:", tokens)
        print("Decoded text:", self.encoding.decode(self.token_ids))
        print("Length of tokens:", len(tokens))
        context_window = 4550

        utilization = len(tokens) / context_window * 100

        print(f"Utilization of context window: {utilization:.2f}%")

bf = Tokeniser()
bf.inspect()
bf.decode()