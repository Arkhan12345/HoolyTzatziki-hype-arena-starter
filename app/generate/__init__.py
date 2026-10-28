def generate(kb, lexicon, input_texts, n=1000, report=None):
    from app.generate.select import generate as _generate

    return _generate(kb, lexicon, input_texts, n, report)


__all__ = ["generate"]
