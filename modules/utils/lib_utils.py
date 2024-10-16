from datetime import datetime


def convert_date_string_to_datetime(date_string):
    format_strings = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ]

    for format_string in format_strings:
        try:
            datetime_object = datetime.strptime(date_string, format_string)
            # Convert to offset-naive datetime object
            datetime_object = datetime_object.replace(tzinfo=None)
            return datetime_object
        except ValueError:
            continue

    raise ValueError(
        f"Date string {date_string} could not be parsed with any known format"
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
