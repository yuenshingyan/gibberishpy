from pygibberish.scanner import GibberishScanner


if __name__ == "__main__":
    scanner = GibberishScanner()
    scanner.load_model(path="transition_matrix_2d.tm")
    additive_cum_proba, multiplicative_cum_proba = scanner.scan("ldfjgnkdfjnd")
    print(additive_cum_proba)
    print(multiplicative_cum_proba)

    # 0.00022810218978102192
    # 0.0
