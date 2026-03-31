from pysentimiento import create_analyzer

analyzer = create_analyzer(task="sentiment", lang="es")


def pysentimiento_predict(text: str) -> str:
    result = analyzer.predict(str(text))
    label = result.output

    # Normalize to your labels
    if label == "POS":
        return "positive"
    elif label == "NEG":
        return "negative"
    else:
        return "neutral"