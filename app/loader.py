import re
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document

# Every page of these HR policy PDFs repeats the same guidance/confidentiality
# notice and a "<policy> Issue # <n> Effective From <date>" line. Once pages are
# joined into continuous text, this boilerplate lands mid-sentence wherever a
# page break happens to fall, corrupting whatever fact it interrupts (e.g. a
# leave-entitlement number). Stripped before chunking so it can't do that.
_HEADER_BOILERPLATE_RE = re.compile(
    r"THIS DOCUMENT PROVIDES GUIDANCE AND IS EXCLUSIVELY FOR USE BY THE STAFF MEMBERS OF"
    r"\s*(?:MONIT PRIVATE LIMITED\.?)?\s*"
    r"ANY ACT OF DIVULGENCE SHALL BE VIEWED VERY SERIOUSLY AND SHALL WARRANT DISCIPLINARY"
    r"(?:\s*ACTION\.?)?",
    re.IGNORECASE,
)
_ISSUE_EFFECTIVE_RE = re.compile(
    r"Issue\s*#\s*\d*\s*Effective From\s+[A-Za-z]+\s+\d{1,2},\s*\d{4}",
    re.IGNORECASE,
)

# The document title usually appears right next to the Issue#/Effective From
# line above and gets removed with it, but sometimes lands on its own (e.g. a
# page-break split the two apart before extraction). Left alone, an orphaned
# title is short enough that it can end up sitting between two genuinely
# unrelated sentences without properly separating them - worse than not
# stripping it. Only matched when isolated on blank-padded lines (never after
# "the ", which is how it's referred to in real prose, e.g. "the Leave Policy
# is to give provision...").
_ORPHAN_TITLE_RE = re.compile(
    r"(?<!the )(?:Leave Policy|Work From Home Policy|Attendance Policy|"
    r"Employee Conduct (?:&|and) Compliance Policy)\s*\n(?:[ \t]*\n)+",
    re.IGNORECASE,
)

# These PDFs extract bullet points as a literal "" glyph with no
# paragraph break before it. RecursiveCharacterTextSplitter prefers to split on
# "\n\n" before falling back to smaller separators, so forcing one in front of
# every bullet nudges chunk boundaries toward landing between bullets instead
# of mid-bullet - it can't fix content that pypdf extracted out of visual
# order in the first place, but it stops the splitter from being indifferent
# to bullet boundaries it otherwise has no signal for.
_BULLET_RE = re.compile(r"[ \t]*")


def _clean_page_text(text: str) -> str:
    text = _HEADER_BOILERPLATE_RE.sub(" ", text)
    text = _ISSUE_EFFECTIVE_RE.sub(" ", text)
    text = _ORPHAN_TITLE_RE.sub(" ", text)
    text = _BULLET_RE.sub("\n\n", text)
    return text


def loading_documents():
    loader = DirectoryLoader(
        "./data", glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    pages = loader.load()

    # PyPDFLoader returns one Document per page, and split_documents() never
    # merges across Document boundaries - so a sentence spanning a page break
    # was always guaranteed to be split, not just unlucky about where a chunk
    # boundary landed. Combine each source file's pages into one continuous
    # Document before chunking so boundaries are only ever determined by
    # chunk_size/overlap, never forced at every page seam.
    merged_by_source = {}
    for page in pages:
        source = page.metadata["source"]
        cleaned = _clean_page_text(page.page_content)
        if source in merged_by_source:
            merged_by_source[source].page_content += "\n\n" + cleaned
        else:
            merged_by_source[source] = Document(page_content=cleaned, metadata={"source": source})

    return list(merged_by_source.values())
