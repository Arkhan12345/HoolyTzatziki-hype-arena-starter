def generate(kb, lexicon, input_texts, n=1000, report=None, **kwargs):
    from app.generate.select import generate as _generate

    return _generate(kb, lexicon, input_texts, n, report, **kwargs)


def generate_from_inventory(inventory, input_texts, n=1000, report=None):
    from app.generate.select import generate_from_inventory as _g

    return _g(inventory, input_texts, n, report)


__all__ = ["generate", "generate_from_inventory"]
