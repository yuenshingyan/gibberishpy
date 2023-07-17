from pygibberish.scanner import GibberishScanner


if __name__ == "__main__":
    scanner = GibberishScanner()
    scanner.build_model(corpus_path=r"C:\Users\Hindy\PycharmProjects\pygibbersh\corpus\corpus.txt", n_gram_size=2)
    scanner.save_model("transition_matrix_2d.tm", encoding="utf-8")
