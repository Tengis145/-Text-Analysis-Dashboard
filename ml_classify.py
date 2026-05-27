"""
ML-based classifiers for Mongolian classroom discourse analysis.

Classifiers (in priority order):
1. BloomBertClassifier  — XLM-RoBERTa embeddings + logistic regression (BloomBERT-style).
                          Downloads ~280 MB on first use; cached by HuggingFace thereafter.
2. BloomClassifier      — keyword matching + TF-IDF cosine fallback (sklearn only, fast).
3. SupportClassifier    — logistic regression binary classifier (teacher response support).
"""
import re
import json
from pathlib import Path
import numpy as np

_AUTOLABEL_FILE = Path(__file__).parent / 'autolabel_cache.json'

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False

# XLM-RoBERTa: strong multilingual model, supports Mongolian out of the box.
# Change to 'bert-base-multilingual-cased' for a lighter (~177 MB) alternative.
BLOOM_BERT_MODEL = 'xlm-roberta-base'

BLOOM_LABELS = {
    'Q1': 'Q1: Сэргээн санах (Remember)',
    'Q2': 'Q2: Ойлгох (Understand)',
    'Q3': 'Q3: Хэрэглэх (Apply)',
    'Q4': 'Q4: Задлан шинжлэх (Analyze)',
    'Q5': 'Q5: Үнэлэх (Evaluate)',
    'Q6': 'Q6: Бүтээх (Create)',
}

BLOOM_COLORS = {
    'Q1': '#3498db', 'Q2': '#2ecc71', 'Q3': '#e67e22',
    'Q4': '#9b59b6', 'Q5': '#e74c3c', 'Q6': '#1abc9c',
}

# ── Official taxonomy keyword roots ────────────────────────────────────────
# Source: Anderson & Krathwohl (2001) revised taxonomy, Mongolian adaptation
# Substring-matched against lowercased, punctuation-stripped text.
# Q2 / Q4 share "харьцуулах" — assigned Q2 since classroom baseline is understanding.
BLOOM_KEYWORDS = {
    'Q1': [  # Сэргээн санах — recall, name, list, identify, observe
        'нэрлэ',          # нэрлэх – name
        'жагсаалт',       # жагсаалт гаргах – list
        'жагсаа',         # жагсаах – list
        'тодорхойл',      # тодорхойлох – identify/define
        'ялгаж тан',      # ялгаж таних – recognize
        'сэргээн сана',   # сэргээн санах – recall
        'байрлуул',       # байрлуулах – locate/place
        'олж хар',        # олж харах – find and see
        'санана уу',      # remember? (recall prompt)
        'давтана уу',     # repeat/recall
        'нэрийг нь хэл',  # say the name of
        'ажигл',          # ажиглах – observe
        'хэд байна',      # how many – factual counting/recall
        'хэдэн байна',    # how many
        'дарааллыг',      # дарааллыг хэлэх – state the sequence
        'хаана байд',     # хаана байдаг – where is it located
        'хэзээ болс',     # хэзээ болсон – when did it happen
        'хэн бэ',         # who is it – factual recall
        'юу вэ',          # what is it – simple factual
        'баримт',         # баримт – fact/evidence (recall)
        'тэмдэглэ',       # тэмдэглэх – note/mark
        'сонго',          # сонгох – select/choose (recognition)
        'заа',            # заах – point out/indicate
        'олно уу',        # олох – find (factual)
        'харна уу',       # харах – look at/see
        # ── нэмэлт keyword ──────────────────────────────────────
        'уншина уу',      # уншиж харах – read aloud (recall)
        'хэлнэ үү',       # хэлэх – say/state (factual)
        'юу юу',          # юу юу байдаг – what are the things
        'ямар ямар',      # ямар ямар зүйл – what kinds
        'хэдэн',          # хэдэн – how many (counting)
        'хаана',          # хаана – where (location recall)
        'хэзээ',          # хэзээ – when (time recall)
        'дахиад нэг',     # дахиад нэг хэлнэ үү – repeat once more
        'юу байна',       # юу байна – what is there
        'харж байна',     # харж байна уу – are you looking
        'нэг нэгээр',     # нэг нэгээр нь – one by one (recall list)
        'анхны',          # анхны алхам – first step (sequence)
        'сүүлийн',        # сүүлийн – last (sequence recall)
        'хэчнээн',        # хэчнээн – how many/much
    ],
    'Q2': [  # Ойлгох — explain, classify, compare, summarize
        'тайлбарл',       # тайлбарлах – explain
        'яагаад',         # why – comprehension question word
        'ялгаа нь',       # what is the difference
        'юугаараа ялгар', # how does it differ
        'ангил',          # ангилах – classify
        'бүлэгл',         # бүлэглэх – group
        'харьцуул',       # харьцуулах – compare
        'жишээ тат',      # жишээ татах – give example
        'жишээ гарга',    # жишээ гаргах – give example
        'өөрийн үгээр',   # in own words
        'дүгнэлт',        # дүгнэлт хийх – conclude
        'дүгн',           # дүгнэх – summarize
        'ойлгосон',       # did you understand
        'ямар учир',      # what reason
        'юу гэсэн үг',    # what does it mean
        'ямар холбоо',    # what connection
        'хэрхэн болд',    # хэрхэн болдог – how does it work
        'утга нь',        # утга нь юу вэ – what is the meaning
        'агуулга',        # агуулга – content/meaning
        'илэрхийл',       # илэрхийлэх – express/represent
        'тайлал',         # тайлал – interpretation
        'ойлгоно уу',     # do you understand
        'зүйрлэ',         # зүйрлэх – analogize
        'хураангуйл',     # хураангуйлах – summarize briefly
        'хэрхэн ойлг',    # хэрхэн ойлгосон – how did you understand
        # ── нэмэлт keyword ──────────────────────────────────────
        'яагаад гэвэл',   # because – explanation marker
        'учир нь',        # учир нь – the reason is
        'ямар утга',      # ямар утгатай – what meaning
        'хэрхэн',         # хэрхэн – how (explanation)
        'ижил тал',       # ижил тал – similarity
        'ялгаатай',       # ялгаатай – different
        'гол санаа',      # гол санаа – main idea
        'тайлбараарай',   # тайлбараарай – please explain
        'ойлгосноо',      # ойлгосноо – what you understood
        'жишээ',          # жишээ – example (broad)
        'ямар нөлөө',     # ямар нөлөө – what effect
        'юуг илэрхийл',   # юуг илэрхийлдэг – what does it express
        'гэж ойлгож',     # гэж ойлгож байна уу – do you understand as
        'хэрхэн тайлбар', # хэрхэн тайлбарлах – how to explain
    ],
    'Q3': [  # Хэрэглэх — apply, implement, calculate, solve
        'хэрэгжүүл',      # хэрэгжүүлэх – implement
        'хувирган',       # хувирган хэрэглэх – apply/transform
        'гүйцэтгэ',       # гүйцэтгэх – execute
        'ашигл',          # ашиглах – use/apply
        'бодлого бод',    # solve a problem
        'тооцоол',        # тооцоолох – calculate
        'ямар аргаар',    # what method (applying a technique)
        'ямар томьёо',    # what formula
        'томьёогоор',     # using formula
        'шийдвэрл',       # шийдвэрлэх – solve/decide
        'дадлага',        # дадлага – practice
        'хэрэгл',         # хэрэглэх – use (broad apply)
        'практик',        # практик – practical application
        'хэрэгжил',       # хэрэгжилт – implementation
        'биелүүл',        # биелүүлэх – fulfill/carry out
        'туршиж үз',      # туршиж үзэх – try out/experiment
        'үйлдл',          # үйлдэл – action/operation
        'хэрэглэж',       # хэрэглэж харуул – demonstrate use
        'практикт',       # практикт ашиглах – use in practice
        'хийж үзэ',       # хийж үзэх – try doing
        'үр дүн',         # үр дүн олох – find the result
        # ── нэмэлт keyword ──────────────────────────────────────
        'хийнэ үү',       # хийнэ үү – please do (action)
        'бодно уу',       # бодно уу – please solve
        'хэрхэн хийх',    # хэрхэн хийх – how to do
        'алхмуудыг',      # алхмуудыг дага – follow the steps
        'дүрмийг',        # дүрмийг хэрэглэ – apply the rule
        'аргыг',          # аргыг хэрэглэ – use the method
        'тооцоог',        # тооцоог хий – do the calculation
        'даалгавар',      # даалгавар – task/assignment
        'хэрэглэн',       # хэрэглэн – applying/using
        'шийдэл',         # шийдэл олох – find solution
        'хийж харуул',    # хийж харуул – show by doing
        'хийцгээе',       # хийцгээе – let's do it
        'бодоод',         # бодоод харна уу – try solving
        'гаргаж тооцо',   # гаргаж тооцоол – calculate and produce
    ],
    'Q4': [  # Задлан шинжлэх — analyze, break down, relate, structure
        'задл',           # задлах – break down/analyze
        'холбон үз',      # холбон үзэх – relate/connect
        'тойм зурга',     # тойм зургийг гаргах – outline/diagram
        'олж илрүүл',     # олж илрүүлэх – discover
        'бүтэцл',         # бүтэцлэх – structure
        'хэсгүүдэд',      # divide into parts
        'учир шалтгаан',  # cause-effect analysis
        'шинжил',         # шинжилэх – examine/analyze
        'нэгтгэ',         # нэгтгэх – integrate
        'зохион байгуул', # зохион байгуулах – organize/arrange
        'нотло',          # нотлох – prove
        'оюуны зураглал', # mind map
        'хамаарлыг',      # the relationship (analysis context)
        'ялгаж задл',     # ялгаж задлах – differentiate and break down
        'судл',           # судлах – study/research
        'хэрхэн бүтс',    # хэрхэн бүтсэн – how is it structured
        'харилцаа',       # харилцаа холбоо – relationship/connection
        'онцлог',         # онцлог – characteristic/feature (analysis)
        'үндсэн санаа',   # үндсэн санаа – main idea (analysis)
        'хуваарил',       # хуваарилах – distribute/divide
        'ангиллын',       # ангиллын үндэс – basis of classification
        # ── нэмэлт keyword ──────────────────────────────────────
        'хэсэг хэсэг',    # хэсэг хэсгээр – part by part
        'юунаас бүтс',    # юунаас бүтсэн – made of what
        'ямар хэсэг',     # ямар хэсгүүдэд – what parts
        'холбоо нь',      # холбоо нь юу вэ – what is the link
        'шалтгааныг',     # шалтгааныг тодорхойл – identify cause
        'үр дагавар',     # үр дагавар – consequence/effect
        'хэв маяг',       # хэв маяг – pattern
        'нотолгоог',      # нотолгоог гарга – provide evidence
        'бүтцийг',        # бүтцийг задал – break down structure
        'дотоод',         # дотоод бүтэц – internal structure
        'ялгаж авч',      # ялгаж авч үзэх – distinguish and examine
        'хамаарал',       # хамаарал – relationship
        'тодорхой болго',  # тодорхой болгох – clarify/specify
    ],
    'Q5': [  # Үнэлэх — evaluate, judge, critique, validate
        'шалга',          # шалгах – check/verify
        'таамагл',        # таамаглах – hypothesize
        'шүүмжил',        # шүүмжлэх – critique
        'баталгаажуул',   # баталгаажуулах – validate
        'хяналт хий',     # хяналт хийх – monitor/control
        'нээн илрүүл',    # нээн илрүүлэх – discover/reveal
        'үнэл',           # үнэлэх – evaluate
        'зөвтг',          # зөвтгөх – justify
        'аль нь илүү',    # which is better
        'зөв үү буруу',   # right or wrong (judgment)
        'хэрхэн сайжруул',# how to improve
        'турши',          # турших – test
        'эргэцүүл',       # эргэцүүлэх – reflect
        'үндэслэ',        # үндэслэх – base on / ground
        'дүгнэж үнэл',    # дүгнэж үнэлэх – conclude and evaluate
        'ямар үндэслэл',  # ямар үндэслэлтэй – what is the rationale
        'санал дүгнэлт',  # санал дүгнэлт – opinion/conclusion
        'ач холбогдол',   # ач холбогдол – significance (evaluative)
        'давуу тал',      # давуу тал – advantage
        'сул тал',        # сул тал – weakness/disadvantage
        'хэр зэрэг',      # хэр зэрэг – to what extent (judgment)
        'нотолгоо',       # нотолгоо – evidence/proof
        # ── нэмэлт keyword ──────────────────────────────────────
        'таны бодол',     # таны бодол – your opinion
        'санал бодол',    # санал бодол – opinion
        'зөв гэж',        # зөв гэж үзэх – consider correct
        'тохиромжтой',    # тохиромжтой эсэх – whether appropriate
        'үр нөлөө',       # үр нөлөө – effectiveness
        'хэр үр дүнтэй',  # хэр үр дүнтэй – how effective
        'шалгуур',        # шалгуур – criterion
        'гэж боддог уу',  # гэж боддог уу – do you think
        'хэр чухал',      # хэр чухал – how important
        'чанарыг',        # чанарыг үнэл – evaluate quality
        'зөв эсэхийг',    # зөв эсэхийг шалга – check correctness
        'нотлох',         # нотлох – prove/substantiate
        'дүгнэж хэл',     # дүгнэж хэл – conclude and state
        'хамгийн',        # хамгийн сайн – the best (superlative judgment)
    ],
    'Q6': [  # Бүтээх — create, design, plan, produce
        'бүтээ',          # бүтээх – create
        'зохион бүтээ',   # зохион бүтээх – design
        'төлөвл',         # төлөвлөх – plan
        'боловсруул',     # боловсруулах – develop/elaborate
        'санаачил',       # санаачлах – initiate/originate
        'загвар гарга',   # загвар гаргах – create a model
        'шинэ зүйл',      # new thing (creation context)
        'санал гарга',    # санал гаргах – propose
        'найруул',        # найруулах – compose/direct
        'програм бич',    # write program
        'зохио',          # зохиох – compose/write (creative)
        'зохиомж',        # зохиомж – composition/design
        'шинэчл',         # шинэчлэх – innovate/renew
        'өөрчл',          # өөрчлөх – modify/transform (creative)
        'гаргаж ир',      # гаргаж ирэх – produce/bring out
        'нэгтгэж бүтээ',  # нэгтгэж бүтээх – combine and create
        'хэрэгжүүлэх төл',# хэрэгжүүлэх төлөвлөгөө – implementation plan
        'шинэ арга',      # шинэ арга – new method
        'бүтээгдэхүүн',   # бүтээгдэхүүн – product (creation)
        'загвар зохио',   # загвар зохиох – design a model
        # ── нэмэлт keyword ──────────────────────────────────────
        'өөрийн загвар',  # өөрийн загвар – own design
        'шинэ санаа',     # шинэ санаа – new idea
        'бүтээж гарга',   # бүтээж гаргах – create and produce
        'зохиогоорой',    # зохиогоорой – please compose
        'дизайн',         # дизайн – design (loanword)
        'хувилбар',       # хувилбар – version/alternative
        'бий болго',      # бий болгох – bring into existence
        'нэгтгэн',        # нэгтгэн гарга – combine and produce
        'гарын авлага',   # гарын авлага – guide/manual (creation)
        'төсөл',          # төсөл – project (creative work)
        'шийдэл олж',     # шийдэл олж бүтээ – find and create solution
        'оригинал',       # оригинал – original (creation)
        'бүтээлч',        # бүтээлч – creative
        'зохиомжил',      # зохиомжилох – design/compose
    ],
}

# ── Bloom training examples — used only for TF-IDF fallback centroid ──────
BLOOM_TRAIN = {
    'Q1': [
        # Бодит ангийн өрөөний ярианаас (ажиглах / санах)
        'Зурагуудаа маш сайн ажиглаад хараарай',
        'Ажигласан зүйлээ ангилаад дэвтэртээ бичээрэй',
        'Эхлээд сайн ажиглана харж байгаад',
        'Багшийн гарт байгаа зүйлийг хар ажиглая',
        'Одоо хичээлдээ анхаарна шүү хүүхдүүдээ',
        'Гараа өргөж байгаад хэлнэ үү',
        'Зүгээр дотроо ажиглаад харъя',
        'Амьд бие хооллодог гэсэн санаа тиймээ',
        'Хөдөлдөг хөдөлдөггүй зүйлийг мэдэж авсан байна',
        'Өсдөгүүдийг нь наасан байна тийм ээ',
        'Амьтай амьгүй бие хоёроо ялгасан байна тиймээ',
        'Хоол иддэг хоол иддэггүй томордог чухал санаа байна',
        # Санах / нэрлэх асуулт
        'Нэрлэнэ үү', 'Жагсаалт гаргана уу', 'Жагсааж бичнэ үү',
        'Тодорхойлно уу', 'Ялгаж таньна уу', 'Сэргээн санана уу',
        'Байрлуулна уу', 'Олно уу', 'Олж харна уу', 'Нэрийг нь хэлнэ үү',
        'Энэ юу вэ', 'Ямар нэртэй вэ', 'Хаана байдаг вэ', 'Хэн бэ',
        'Хэзээ болсон бэ', 'Юу юу байдаг вэ', 'Давтана уу', 'Санана уу',
        'Дарааллыг хэлнэ үү', 'Баримтыг нэрлэнэ үү', 'Тэмдэглэнэ үү',
        'Ямар зүйлс байдаг вэ', 'Жагсаасан байгаарай', 'Хэдэн байна вэ',
        'Энд юу байна вэ', 'Зурган дээр юу харагдаж байна вэ',
        'Энэ юу юм бэ дүрсийг нэрлэнэ үү',
        'Тэмдэглэгээг нэрлэнэ үү', 'Ямар өнгөтэй вэ',
        'Хэчнээн байна вэ', 'Юуг ажиглаж байна вэ',
        'Хаана байрладаг вэ', 'Дарааллыг заана уу', 'Анхны алхам нь юу вэ',
        'Хамгийн эхнийх нь юу вэ', 'Сүүлийн нь юу вэ',
        'Эргэж санана уу', 'Өмнө нь ярьсныг санана уу',
        'Жагсааж хэлнэ үү', 'Нэг нэгээр нь нэрлэнэ үү',
        'Ямар баримт байна вэ', 'Хэн хэн оролцсон бэ',
        'Юу болсон бэ', 'Энэ хаана харагддаг вэ',
        'Ямар нэр өгдөг вэ', 'Онцлогийг жагсаана уу',
        'Хоёр юм нэрлэнэ үү', 'Гурван жишээ нэрлэнэ үү',
        # ── нэмэлт жишээ ────────────────────────────────────────
        'Уншина уу', 'Хэлнэ үү', 'Хаана байдаг болохыг хэлнэ үү',
        'Дахиад нэг хэлнэ үү', 'Дахиад давтана уу',
        'Юу юу байдгийг нэрлэнэ үү', 'Ямар ямар зүйл байдаг вэ',
        'Тоолно уу', 'Хэдэн байдаг вэ', 'Хичнээн байна вэ',
        'Энэ юу вэ нэрийг нь хэлнэ үү', 'Нэрийг хэлнэ үү',
        'Хэн хийсэн бэ', 'Хэзээ хийсэн бэ', 'Хаанаас ирсэн бэ',
        'Юу харж байна вэ', 'Энд юу харагдаж байна вэ',
        'Ямар зүйл байна вэ', 'Юуг олж харсан бэ',
        'Дарааллаар нэрлэнэ үү', 'Эхний нь юу вэ',
        'Хоёр дахь нь юу вэ', 'Гурав дахь нь юу вэ',
        'Ямар хэлбэртэй вэ', 'Ямар хэмжээтэй вэ',
        'Хэдэн ширхэг байна вэ', 'Нийт хэдэн байна вэ',
        'Юуг санаж байна вэ', 'Энэ тухай юу мэдэх вэ',
        'Гараа өргөнө үү', 'Хэн мэдэх вэ',
        'Хэн хэлэх вэ', 'Хэн нэрлэх вэ',
    ],
    'Q2': [
        # Бодит ярианаас (тайлбарлах / ангилах / харьцуулах)
        'Яагаад амьд бус гэж ангилсан юм бол ялгаа нь юу вэ',
        'Амьд ба амьд бус зүйлсийн ялгааг ярилцъя ялгаа нь юу юм бол',
        'Энэ хэсгийг юугаар нь ингэж ангилсан бэ',
        'Амьд бие нь амьд бусаасаа юугаараа ялгардаг юм бол',
        'Хоёулаа ярьсан зүйлээ юу нь ижил байгаа юу нь ялгаатай байгаа',
        'Яаж загварчилмаар байна ямар загварт оруулж ангилмаар байна',
        'Дараагийн ангилалд юу гэх юм бол',
        'Амьд биеэ дахиад ангилъя дотор нь ангилна ямар шинж чанараар',
        'Туулайг хурдан яст мэлхий удаан гэж ангилсан байна',
        'Цэцэг шар мод шар биш гэж ангилсан байна',
        'Хүүхэд хөдөлдөг нар доор хөдөлдөг гэж ангилсан байна',
        'Амьд амьд бус зүйлс гэж хоёр ангилсан байна',
        'Байгаль буюу хот гэж ангилсан байна',
        'Хүн амьтан ургамал гэж ангилж болох юм байна',
        'Ургамал гэж ангилж болох юм байна дахиад нэг ангиллаа давтъя',
        'Өсдөг өсдөггүй гэж наасан байна ялгаа нь юу вэ',
        'Маш олон янзаар ангилсан байна амьд амьд бус хурдан удаан',
        'Өөр өөрсдийнхөө санаагаар ангиллаа биччихсэн тийм ээ',
        # Ойлгох / тайлбарлах
        'Тайлбарлана уу', 'Дүгнэнэ үү', 'Дүгнэлт хийнэ үү',
        'Өөрийн үгээр илэрхийлнэ үү', 'Өөрийн үгээр хэлнэ үү',
        'Ангилна уу', 'Бүлэглэнэ үү', 'Харьцуулна уу',
        'Жишээ татана уу', 'Жишээ гаргана уу', 'Тодотгож харуулна уу',
        'Ямар утгатай вэ', 'Яагаад гэж', 'Юу гэсэн үг вэ',
        'Хэрхэн ойлгосон бэ', 'Ялгаа нь юу вэ', 'Ижил тал нь юу вэ',
        'Ямар учиртай вэ', 'Хэрхэн болдог вэ', 'Ойлгосноо хэлнэ үү',
        'Зүйрлэж тайлбарлана уу', 'Агуулгыг ойлгосноо харуулна уу',
        'Юу гэсэн утгатай вэ', 'Яагаад ийм байна вэ',
        'Яагаад тийм болдог вэ', 'Хэрхэн ойлгох вэ',
        'Ойлгосоноо тайлбарлаарай', 'Өөрийн үгээр тайлбарлаарай',
        'Хоёрыг харьцуулна уу', 'Ямар ялгаатай вэ',
        'Ямар ижил талтай вэ', 'Учир нь юу вэ',
        'Яагаад чухал вэ', 'Ямар холбоотой вэ',
        'Ямар утга агуулдаг вэ', 'Тайлбарлаж өгнэ үү',
        'Хэрхэн болдогийг тайлбарла', 'Ямар нөлөө үзүүлдэг вэ',
        'Жишээгээр тайлбарлана уу', 'Богинохон дүгнэнэ үү',
        'Гол санаа нь юу вэ', 'Агуулга нь юу вэ',
        'Ойлгосоноо харуулна уу', 'Хэрхэн тайлбарлах вэ',
        'Бусдад хэрхэн тайлбарлах вэ', 'Яагаад гэдгийг тайлбарла',
        # ── нэмэлт жишээ ────────────────────────────────────────
        'Яагаад тийм гэж боддог вэ', 'Яагаад ийм болсон гэж хэлэх вэ',
        'Хэрхэн ойлгоход хялбар вэ', 'Ямар утга илэрхийлж байна вэ',
        'Энэ хоёрын ялгаа юу вэ', 'Юу нь адилхан байна вэ',
        'Гол санааг тайлбарлаарай', 'Товч тайлбарлаарай',
        'Яагаад ийм үр дүн гарсан бэ', 'Ямар учраас тийм болдог вэ',
        'Хэрхэн ойлгож болох вэ', 'Өөрийнхөө үгээр хэлж өгнэ үү',
        'Юуг хэлэхийг оролдож байна вэ', 'Ямар санааг илэрхийлж байна вэ',
        'Тайлбарлаж чадах уу', 'Гол утга нь юу вэ',
        'Харьцуулж тайлбарлаарай', 'Ижил ба ялгаатай талыг хэлнэ үү',
        'Яагаад тийм гэж ангилсан бэ', 'Ямар шинжээр ангилсан бэ',
        'Ямар шалтгаанаар тийм болдог вэ', 'Гэж ойлгож байна уу',
    ],
    'Q3': [
        # Хэрэглэх / бодлого шийдэх
        'Хэрэгжүүлнэ үү', 'Хувирган хэрэглэнэ үү', 'Гүйцэтгэнэ үү',
        'Ашиглана уу', 'Хэрэглэнэ үү', 'Хэрхэн хэрэглэх вэ',
        'Тооцоолно уу', 'Шийдвэрлэнэ үү', 'Бодлого бодно уу',
        'Томьёогоор тооцоолно уу', 'Ямар томьёо ашиглах вэ',
        'Ямар аргаар бодох вэ', 'Туршиж үзнэ үү', 'Дадлага хийнэ үү',
        'Практикт ашиглана уу', 'Үр дүнг олно уу', 'Хэрхэн шийдэх вэ',
        'Тооцооллыг хийнэ үү', 'Биелүүлнэ үү',
        'Энэ аргыг хэрэгжүүлнэ үү', 'Дүрмийг хэрэглэнэ үү',
        'Жишээн дээр хийж үзнэ үү', 'Бодлогыг шийдвэрлэнэ үү',
        'Аргаа хэрэглэж бодно уу', 'Практик даалгавар хийнэ үү',
        'Хэрхэн хийх вэ гэдгийг харуулна уу',
        'Арга барилыг хэрэглэнэ үү', 'Өмнөх мэдлэгээ ашиглана уу',
        'Тооцооллыг гүйцэтгэнэ үү', 'Алгоритмыг хэрэгжүүлнэ үү',
        'Загварыг хэрэглэнэ үү', 'Шийдлийг олж харуулна уу',
        'Энэ нөхцөлд ямар арга хэрэглэх вэ',
        'Мэдлэгээ шинэ нөхцөлд ашиглана уу',
        'Хэрэглэж болох арга нь юу вэ', 'Хэрэгжүүлнэ үү харуулна уу',
        'Ямар үйлдэл хийх вэ', 'Дараа нь юу хийх вэ',
        'Хэрхэн хийж болохыг харуулна уу',
        # ── нэмэлт жишээ ────────────────────────────────────────
        'Хийнэ үү', 'Бодно уу', 'Хэрхэн хийх вэ харуулна уу',
        'Аргыг хэрэглэж шийдэнэ үү', 'Томьёо ашиглаж тооцоолно уу',
        'Дүрмийг хэрэглэж бодно уу', 'Жишээн дээр туршина уу',
        'Энэ мэдлэгийг ашиглаж шийдвэрлэнэ үү',
        'Алхам алхмаар хийж харуулна уу',
        'Дадлага хийж үзнэ үү', 'Хэрэглэж болохыг харуулна уу',
        'Практикт хэрхэн ашиглах вэ', 'Хийж үзье',
        'Тооцоог гаргана уу', 'Хариуг олно уу',
        'Шийдлийг гаргана уу', 'Хэрхэн гүйцэтгэх вэ',
        'Дараагийн алхам юу вэ', 'Яаж хийх вэ',
        'Аргачлалыг дага', 'Ямар дараалалтай хийх вэ',
        'Хэрхэн бодох вэ', 'Аль аргыг хэрэглэх вэ',
        'Хэрэглэж харуулна уу', 'Туршиж харна уу',
        'Хийцгээе', 'Бодцгооё', 'Хамтдаа шийдье',
    ],
    'Q4': [
        # Задлан шинжлэх / холбох
        'Задлана уу', 'Холбон үзнэ үү', 'Тойм зургийг гаргана уу',
        'Олж илрүүлнэ үү', 'Бүтэцлэнэ үү', 'Нэгтгэнэ үү',
        'Зохион байгуулна уу', 'Шинжилнэ үү', 'Хэсгүүдэд хуваана уу',
        'Ямар хэсгүүдэд бүтсэн вэ', 'Хамаарал нь юу вэ',
        'Ямар шалтгаантай вэ', 'Судлана уу', 'Хэрхэн бүтсэн вэ',
        'Учир шалтгааныг олно уу', 'Нотлоно уу', 'Харьцуулан шинжилнэ үү',
        'Хамаарлыг тодорхойлно уу', 'Оюуны зураглал хийнэ үү',
        'Бүтцийг задлана уу', 'Хэсэг хэсгээр нь авч үзнэ үү',
        'Ямар хэсгүүдээс тогтдог вэ', 'Хоорондын холбоог харуулна уу',
        'Яагаад ийм болсон бэ учрыг тайлбарла',
        'Шалтгаан үр дагаврыг тодорхойлно уу',
        'Нотолгоогоо гаргана уу', 'Дүн шинжилгээ хийнэ үү',
        'Ямар хэв маяг байна вэ', 'Судалгаа хийнэ үү',
        'Бүтцийг шинжилнэ үү', 'Хоорондын хамаарлыг тодорхойл',
        'Элементүүдийг ялгаж задла', 'Дотоод бүтцийг харуулна уу',
        'Хэсгүүд нь хэрхэн холбоотой вэ',
        'Гол ба дэд санааг ялгана уу', 'Холбоог олж харна уу',
        'Хоёрын ялгааг задлан шинжилнэ үү',
        'Нотлох баримт гаргана уу', 'Гүнзгий шинжилгээ хийнэ үү',
        'Ямар хүчин зүйлс нөлөөлдөг вэ',
        # ── нэмэлт жишээ ────────────────────────────────────────
        'Задлан шинжилнэ үү', 'Хэсгүүдэд задлана уу',
        'Юунаас бүрдэж байна вэ', 'Ямар хэсгүүдэд хуваагдах вэ',
        'Холбоог тодорхойлно уу', 'Хамаарлыг задлана уу',
        'Шалтгааныг олж тодорхойлно уу', 'Үр дагаварыг тодорхойлно уу',
        'Бүтцийг нь задалж харуулна уу', 'Дотоод холбоог харуулна уу',
        'Ямар хэв маягтай байна вэ', 'Нотлогдох уу нотолно уу',
        'Хэрхэн зохион байгуулагдсан вэ', 'Ямар бүтэцтэй вэ',
        'Ерөнхий ба тусгай санааг ялга', 'Гол санаа дэд санааг задал',
        'Ямар үндэслэлтэй вэ', 'Хоёр зүйлийг задлан харьцуул',
        'Хэсэг тус бүрийн үүргийг тайлбарла',
        'Элемент бүрийг тодорхойлно уу',
        'Системийн бүтцийг задла', 'Ямар хэсгүүдэд хуваагдах вэ',
        'Учир шалтгааныг тайлбарла', 'Хэрхэн хамааралтай вэ',
    ],
    'Q5': [
        # Үнэлэх / дүгнэх
        'Шалгана уу', 'Таамаглана уу', 'Шүүмжилнэ үү',
        'Туршина уу', 'Шинжлэнэ үү', 'Нээн илрүүлнэ үү',
        'Хяналт хийнэ үү', 'Баталгаажуулна уу', 'Хянаж үзнэ үү',
        'Эргэцүүлнэ үү', 'Үнэлнэ үү', 'Шүүмжлэл хийнэ үү',
        'Ямар байна гэж боддог вэ', 'Зөвтгөнө үү',
        'Хамгийн сайн нь аль вэ', 'Зөв үү буруу үү',
        'Аль нь илүү оновчтой вэ', 'Зөв гэж үзэх үндэслэл нь юу вэ',
        'Хэрхэн сайжруулах байсан бэ', 'Санал бодлоо хэлнэ үү',
        'Энэ зөв үү буруу үү яагаад', 'Ямар давуу сул талтай вэ',
        'Хамгийн оновчтой нь аль вэ', 'Таны үнэлгээ ямар байна вэ',
        'Энэ хариулт зөв гэж үзэх үү', 'Шүүмжлэлт дүгнэлт хийнэ үү',
        'Хэр зэрэг үр дүнтэй вэ', 'Ямар шалгуурыг ашиглах вэ',
        'Нотолгоо дээр үндэслэн дүгнэнэ үү',
        'Үнэ цэнийг тодорхойлно уу', 'Хэр чухал вэ яагаад',
        'Аль арга нь илүү тохиромжтой вэ',
        'Шийдвэр гаргахдаа ямар шалгуур ашиглах вэ',
        'Дүгнэж хэлнэ үү зөв гэж үзэх үү',
        'Энэ арга нь хэр үр дүнтэй вэ',
        'Ямар нотолгоо дээр тулгуурласан вэ',
        'Сайжруулах санал гаргана уу', 'Дахин эргэцүүлж үзнэ үү',
        'Ямар дүгнэлтэд хүрч болох вэ',
        'Өөрийнхөө хариултыг шалгана уу',
        # ── нэмэлт жишээ ────────────────────────────────────────
        'Таны бодлоор аль нь зөв вэ', 'Хэр зэрэг найдвартай вэ',
        'Зөв эсэхийг шалгана уу', 'Шалгуурын үүднээс дүгнэнэ үү',
        'Ямар үндэслэлтэй бол', 'Аль илүү тохиромжтой вэ',
        'Хэр чухал ач холбогдолтой вэ', 'Гэж боддог уу яагаад',
        'Сул ба давуу талыг хэлнэ үү', 'Сул тал нь юу вэ',
        'Давуу тал нь юу вэ', 'Хамгийн тохиромжтой нь аль вэ',
        'Нотолгоог дүгнэнэ үү', 'Ямар шийдвэр гаргах вэ',
        'Хэр зэрэг үнэн бэ', 'Батлагдсан уу эсэхийг шалгана уу',
        'Хамгийн сайн шийдэл аль вэ', 'Үр дүнг үнэлнэ үү',
        'Зөв буруу эсэхийг тогтоо', 'Аль нь илүү дээр вэ',
        'Хэрхэн сайжруулж болох вэ', 'Ямар байвал илүү сайн байх вэ',
        'Санал дүгнэлтээ хэлнэ үү', 'Шүүн дүгнэнэ үү',
    ],
    'Q6': [
        # Бүтээх / зохион бүтээх / төлөвлөх
        'Бүтээнэ үү', 'Зохионо уу', 'Төлөвлөнэ үү',
        'Боловсруулна уу', 'Санаачилна уу', 'Зохион бүтээнэ үү',
        'Төлөвлөгөө гаргана уу', 'Загвар гаргана уу', 'Шинэ зүйл зохионо уу',
        'Хэрхэн сайжруулах вэ', 'Ямар арга боловсруулах вэ',
        'Санал гаргана уу', 'Шинэ аргаар бүтээнэ үү',
        'Шинэ санаа оруулна уу', 'Загвар зохиона уу', 'Найруулна уу',
        'Шинэ зүйл гаргана уу', 'Програм бичнэ үү',
        'Өөрийн санааг бүтээнэ үү', 'Шинэ бүтээл гаргана уу',
        'Зохион бүтээнэ үү харуулна уу', 'Бодлогыг шинэ аргаар шийдэнэ үү',
        'Өөрийн загвар зохиона уу', 'Шинэ систем боловсруулна уу',
        'Бүтээлч ажил хийнэ үү', 'Оригинал бүтээл гаргана уу',
        'Шинэ арга зам санаачилна уу', 'Нийлүүлж шинэ зүйл бүтээнэ үү',
        'Шинэ дизайн гаргана уу', 'Гарын авлага боловсруулна уу',
        'Төслийн санал бичнэ үү', 'Ямар шинэ шийдэл байж болох вэ',
        'Шинэ хувилбар зохиона уу', 'Бүтээгдэхүүн гаргана уу',
        'Яаж шинэчилж болох вэ', 'Хэрэглэгчдэд зориулсан загвар гаргана уу',
        'Шинэ аргачлал боловсруулна уу',
        'Үр дүнтэй шийдэл бүтээж харуулна уу',
        # ── нэмэлт жишээ ────────────────────────────────────────
        'Өөрийн санаагаар бүтээнэ үү', 'Шинэ зүйл гаргаж ирнэ үү',
        'Зохиож бичнэ үү', 'Загвар зохионо уу',
        'Шинэ арга санаачилна уу', 'Боловсруулж гаргана уу',
        'Бүтээлч байдлаар шийдэнэ үү', 'Оригинал загвар хийнэ үү',
        'Өөрийн загварыг хийнэ үү', 'Шинэ дизайн хийнэ үү',
        'Шинэ төлөвлөгөө гаргана уу', 'Шинэ хэлбэр зохионо уу',
        'Багаараа бүтээл хийнэ үү', 'Бие даан зохионо уу',
        'Шинэ бүтэц боловсруулна уу', 'Ямар шинэ зүйл хийж болох вэ',
        'Санаагаа нэгтгэж бүтээнэ үү', 'Хэрэглэгчдэд хэрэгтэй зүйл хийнэ үү',
        'Шинэ хандлагаар хийнэ үү', 'Бүтээлч шийдэл гаргана уу',
        'Өөрийн санааг хэрэгжүүлнэ үү', 'Шинэ загвар бий болгоно уу',
        'Туршилт хийж шинэ зүйл бүтээнэ үү',
        'Өмнөхөөс өөр арга хэрэглэж бүтээнэ үү',
    ],
}

# ── Support classifier training data — full Mongolian classroom sentences ─────
# Дэмжсэн: teacher affirms/repeats/praises student's answer
# Дэмжээгүй: teacher gives instruction, moves on, or manages class without acknowledging
SUPPORT_POS = [
    # Explicit praise
    'Маш сайн хариулсан байна. Яг тийм байна.',
    'Зөв байна. Маш сайн.',
    'Тийм ээ, маш сайн байна.',
    'Гоё тайлбарласан байна. Баярлалаа.',
    'Яг зөв байна. Чи сайн бодсон байна.',
    'Маш сайн ажигласан байна. Тийм ээ.',
    'Зөв ойлгосон байна. Сайхан тайлбарлав.',
    'Оновчтой байна. Гоё.',
    'Тийм шүү. Сайн байна.',
    'Ухаалаг хариулт байна.',
    'Зөв зөв байна. Тиймээ.',
    'Гоё санасан байна.',
    'Үнэн байна. Сайн тайлбарлалаа.',
    'Чадварлаг хариулсан байна. Баярлалаа.',
    'Тийм л дээ. Зөв байна.',
    'Маш сайн байна баярлалаа.',
    'Болж байна. Яг тийм.',
    'Маш сайн сонсоно шүү.',
    'Та хоёр хөөрхөн ангилсан байсан шүү дээ.',
    'Маш олон янзаар ангилсан байна тиймээ.',
    # Repeat/paraphrase student's answer — supportive validation
    'Амьд бие хооллодог гэсэн санаа. Тиймээ.',
    'Туулайг хурдан гэж ангилсан байна. Тийм ээ.',
    'Хөдөлдөг учраас тиймээ. Зөв байна.',
    'Өсдөг өсдөггүй гэж наасан байна. Маш сайн.',
    'Амьтан гэж ангилж болох юм байна. Тиймээ.',
    'Ургамал гэж ангилж болох юм байна.',
    'Хүн амьтан ургамал гэж ангилсан байна. Тиймээ.',
    'Цэцэг шар гэж ангилсан байна. Зөв байна.',
    'Маш сайн байна. Бусад нь сонссон уу?',
    'Зөв санаа байна. Тиймээ. Маш сайн.',
    # Building on student answer
    'Зөв байна. Тэгвэл бусдыгаа сонсъё.',
    'Тиймээ. Нэмж хэлэх хүн байна уу?',
    'Сайн байна. Энэ санааг дэмждэг хэн байна?',
    'Зөв хариулсан байна. Тэгвэл дараагийн нь.',
    'Маш сайн. За бусдыгаа сонсъё.',
]

SUPPORT_NEG = [
    # Classroom management — no acknowledgment of student answer
    'Одоо хичээлдээ анхаарна шүү хүүхдүүдээ.',
    'Зөв суугаарай. Анхаарна уу.',
    'Урагшаа харна уу. Хосоороо ярилцаарай.',
    'Анхаарна уу. Бичнэ үү.',
    'Дуусгаарай. Хугацаа дуусах гэж байна.',
    'Сонс сонс. Анхааралтай сонсоорой.',
    'Бүгд анхаарна уу. Цааш явцгаая.',
    # Moving to next activity/student without acknowledging
    'Одоо дараагийн асуулт руу орцгооё.',
    'Тэгвэл дараагийн хүнийг сонсъё.',
    'За Сувдынхыг сонсъё. Дараагийн ээлжинд.',
    'Одоо бид дараагийн ажил руу орно.',
    'Цааш үргэлжлүүлцгээе. Тэгвэл дараагийн.',
    'За ерөнхийдөө ийм байна. Тэгвэл дараагийнх.',
    'Хэн хариулах вэ? Бусад нь мэдэх үү?',
    'Тэгвэл нэг минут байна шүү. Дуусгаарай.',
    # Instructions without acknowledging
    'Ажигласан зүйлээ ангилаад дэвтэртээ бичээрэй.',
    'Амьд биеэ дахиад ангилъя. Дотор нь ангилна.',
    'За багш нь даалгаврыг нь дахиад нэг хэлээд өгье.',
    'Одоо хамтдаа ажиллацгаая. Дуусгана уу.',
    'Уншина уу. Тэгээд хариулна уу.',
    'Тэгвэл эндээс харж байгаад тайлбарлаарай.',
    # Correction or redirection — no validation
    'Буруу байна. Дахиад бодно уу.',
    'Алдаатай байна. Дахин хийнэ үү.',
    'Тийм биш. Өөр хариулт байна уу?',
    'Дахиад нэг бодоод харна уу.',
    'Чанга хэлээрэй. Дахин хэлнэ үү.',
    'Ойлгосонгүй юм байна. Нэг дахин хэлнэ.',
    'Хэлж чадахгүй байна уу? Бусад нь.',
    'Нэг удаа давтана уу. Анхааралтай сонсоорой.',
    'Дахиад нэг удаа хийнэ үү.',
    'Тэгвэл өөр санаа байна уу?',
]


def _preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text


class BloomBertClassifier:
    """BloomBERT-style classifier for Mongolian classroom questions.

    Encodes text with XLM-RoBERTa (mean-pool of last hidden state), then
    classifies with logistic regression trained on BLOOM_TRAIN examples.
    Model is downloaded once (~280 MB) and cached by HuggingFace.

    Confidence = softmax probability of the predicted class from LR.
    Combined score = 0.6 × BERT_prob + 0.4 × keyword_boost so that
    strong keyword matches still raise confidence.
    """

    def __init__(self, model_name=BLOOM_BERT_MODEL, extra_train=None):
        import warnings, logging
        logging.getLogger('transformers').setLevel(logging.ERROR)
        warnings.filterwarnings('ignore', category=UserWarning, module='huggingface_hub')
        self._classes = list(BLOOM_TRAIN.keys())
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()

        # Merge base training data with any auto-labeled extras
        train_data = {k: list(v) for k, v in BLOOM_TRAIN.items()}
        if extra_train:
            for cls, txts in extra_train.items():
                train_data.setdefault(cls, []).extend(txts)

        texts, labels = [], []
        for cls in self._classes:
            for t in train_data.get(cls, []):
                texts.append(t)
                labels.append(cls)

        emb = self._encode(texts)
        self._lr = LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000,
            random_state=42,
        )
        self._lr.fit(emb, labels)

    def _encode(self, texts, batch_size=16):
        """Return (N, hidden) mean-pooled XLM-R embeddings."""
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors='pt',
            )
            with torch.no_grad():
                out = self._model(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_emb.append(pooled.cpu().numpy())
        return np.vstack(all_emb)

    def _kw_hit(self, text):
        """Return fraction of keyword sets that fire (0–1) for the best class."""
        t = _preprocess(text)
        hits = {cls: sum(1 for kw in BLOOM_KEYWORDS[cls] if kw in t)
                for cls in self._classes}
        best = max(hits.values())
        return hits, best

    def _scores(self, text):
        emb = self._encode([text])
        proba = self._lr.predict_proba(emb)[0]
        bert_probs = {c: float(p) for c, p in zip(self._lr.classes_, proba)}

        kw_hits, best_count = self._kw_hit(text)
        total_kw = sum(kw_hits.values())

        if total_kw > 0:
            kw_probs = {c: kw_hits[c] / total_kw for c in self._classes}
            # Blend: BERT dominates but keyword signal lifts the winner
            combined = {c: 0.65 * bert_probs[c] + 0.35 * kw_probs[c]
                        for c in self._classes}
            kw_boost = min(0.15 * best_count, 0.25)
            best_cls = max(combined, key=combined.get)
            confidence = min(combined[best_cls] + kw_boost, 0.97)
        else:
            combined = bert_probs
            best_cls = max(combined, key=combined.get)
            confidence = combined[best_cls]

        s = sum(combined.values())
        probs = {c: round(v / s, 4) for c, v in combined.items()}
        return best_cls, round(confidence, 3), probs

    def predict(self, text):
        best, conf, _ = self._scores(text)
        return best, conf

    def predict_all(self, text):
        _, _, probs = self._scores(text)
        return probs


class BloomClassifier:
    """Bloom's taxonomy classifier.

    Primary: keyword matching on official Anderson & Krathwohl (2001) Mongolian verb roots.
    Fallback: TF-IDF char n-gram cosine similarity to per-level training centroids.

    Confidence:
      keyword match  → 0.45 + 0.15 × match_count  (capped at 0.92)
      TF-IDF only    → proportional cosine score   (typically 0.15–0.25)
    """

    def __init__(self, extra_train=None):
        self._classes = list(BLOOM_TRAIN.keys())
        train_data = {k: list(v) for k, v in BLOOM_TRAIN.items()}
        if extra_train:
            for cls, txts in extra_train.items():
                train_data.setdefault(cls, []).extend(txts)

        all_texts = [_preprocess(e) for c in self._classes for e in train_data.get(c, [])]
        self._vec = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4),
            sublinear_tf=True, max_features=8000,
        )
        self._vec.fit(all_texts)
        self._centroids = {}
        for cls in self._classes:
            vecs = self._vec.transform([_preprocess(e) for e in train_data.get(cls, [])])
            self._centroids[cls] = np.asarray(vecs.mean(axis=0))

    def _kw_scores(self, text):
        t = _preprocess(text)
        return {cls: sum(1 for kw in BLOOM_KEYWORDS[cls] if kw in t)
                for cls in self._classes}

    def _tfidf_probs(self, text):
        qvec = self._vec.transform([_preprocess(text)])
        raw = {cls: float(cosine_similarity(qvec, c)[0][0])
               for cls, c in self._centroids.items()}
        vals = np.array([raw[c] for c in self._classes])
        s = vals.sum()
        probs = vals / s if s > 0 else np.ones(len(vals)) / len(vals)
        return {c: round(float(p), 4) for c, p in zip(self._classes, probs)}

    def _scores(self, text):
        kw = self._kw_scores(text)
        total_kw = sum(kw.values())
        if total_kw > 0:
            probs = {c: kw[c] / total_kw for c in self._classes}
            best_count = max(kw.values())
            confidence = min(0.45 + 0.15 * best_count, 0.92)
        else:
            probs = self._tfidf_probs(text)
            confidence = max(probs.values())
        return probs, round(confidence, 3)

    def predict(self, text):
        probs, confidence = self._scores(text)
        best = max(probs, key=probs.get)
        return best, confidence

    def predict_all(self, text):
        probs, _ = self._scores(text)
        return {c: round(p, 4) for c, p in probs.items()}


class SupportClassifier:
    """Logistic regression classifier: teacher response supportive (1) or not (0)."""

    def __init__(self):
        X = [_preprocess(t) for t in SUPPORT_POS + SUPPORT_NEG]
        y = [1] * len(SUPPORT_POS) + [0] * len(SUPPORT_NEG)
        self._vec = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4),
            sublinear_tf=True, max_features=3000,
        )
        X_vec = self._vec.fit_transform(X)
        self._model = LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=42
        )
        self._model.fit(X_vec, y)

    def predict(self, text):
        vec = self._vec.transform([_preprocess(text)])
        proba = self._model.predict_proba(vec)[0]
        idx = int(np.argmax(proba))
        label = 'Дэмжсэн' if idx == 1 else 'Дэмжээгүй'
        return label, round(float(proba[idx]), 3)


# ── Singletons ────────────────────────────────────────────────────────
_bloom_bert_clf = None   # BloomBertClassifier (preferred)
_bloom_clf = None        # BloomClassifier (TF-IDF fallback)
_support_clf = None
_bert_failed = False     # set True if BERT model load fails


def _get_bloom():
    """Return best available Bloom classifier: BERT > TF-IDF > None."""
    global _bloom_bert_clf, _bloom_clf, _bert_failed

    if not _bert_failed and TRANSFORMERS_OK and SKLEARN_OK:
        if _bloom_bert_clf is None:
            try:
                _bloom_bert_clf = BloomBertClassifier()
            except Exception:
                _bert_failed = True
        if _bloom_bert_clf is not None:
            return _bloom_bert_clf

    if _bloom_clf is None and SKLEARN_OK:
        _bloom_clf = BloomClassifier()
    return _bloom_clf


def _get_support():
    global _support_clf
    if _support_clf is None and SKLEARN_OK:
        _support_clf = SupportClassifier()
    return _support_clf


def using_bert():
    """Return True if the BERT classifier is active."""
    return isinstance(_get_bloom(), BloomBertClassifier)


def bloom_classify(text):
    """Return (level, confidence) — level is 'Q1'..'Q6'."""
    clf = _get_bloom()
    if clf is None:
        return 'Q1', 0.0
    return clf.predict(text)


def bloom_proba(text):
    """Return {level: probability} dict for all 6 levels."""
    clf = _get_bloom()
    if clf is None:
        return {c: 1/6 for c in BLOOM_TRAIN}
    return clf.predict_all(text)


def support_classify(text):
    """Return ('Дэмжсэн'/'Дэмжээгүй', confidence)."""
    clf = _get_support()
    if clf is None:
        return 'Дэмжээгүй', 0.5
    return clf.predict(text)


# ── Auto-label system ──────────────────────────────────────────────────────────

def _load_autolabels():
    """Load persisted auto-labeled examples from JSON cache."""
    if _AUTOLABEL_FILE.exists():
        with open(_AUTOLABEL_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {c: [] for c in BLOOM_TRAIN}


def _save_autolabels(data):
    with open(_AUTOLABEL_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collect_autolabel(text, threshold=0.70):
    """Classify text; if confidence ≥ threshold, persist as auto-label.

    Returns (level, confidence) same as bloom_classify.
    Call retrain_bloom() periodically to incorporate collected labels.
    """
    level, conf = bloom_classify(text)
    if conf >= threshold:
        data = _load_autolabels()
        bucket = data.setdefault(level, [])
        clean = text.strip()
        # Avoid duplicates (exact match)
        if clean and clean not in bucket and clean not in BLOOM_TRAIN.get(level, []):
            bucket.append(clean)
            _save_autolabels(data)
    return level, conf


def autolabel_stats():
    """Return dict with auto-label counts per level and total."""
    data = _load_autolabels()
    counts = {c: len(data.get(c, [])) for c in BLOOM_TRAIN}
    counts['_total'] = sum(counts.values())
    return counts


def retrain_bloom(min_new=5):
    """Retrain the Bloom classifier with auto-labeled data.

    Only retrains when at least `min_new` new examples have been collected.
    Returns True if retrained, False if not enough data yet.
    """
    global _bloom_bert_clf, _bloom_clf, _bert_failed

    extra = _load_autolabels()
    total_new = sum(len(v) for v in extra.values())
    if total_new < min_new:
        return False

    if not _bert_failed and TRANSFORMERS_OK and SKLEARN_OK:
        try:
            model_name = _bloom_bert_clf._tokenizer.name_or_path if _bloom_bert_clf else BLOOM_BERT_MODEL
            _bloom_bert_clf = BloomBertClassifier(model_name=model_name, extra_train=extra)
            return True
        except Exception:
            _bert_failed = True

    if SKLEARN_OK:
        _bloom_clf = BloomClassifier(extra_train=extra)
        return True

    return False
