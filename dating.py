import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import STOPWORDS
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
from plotly.offline import iplot
from plotly.subplots import make_subplots
import zipfile


def plot_pie_chart_for_value_counts(column):
    fig = go.Figure(data=[go.Pie(labels=column.value_counts().index,
                                 values=column.value_counts().values,
                                 textinfo='label',
                                 textfont_size=15)])
    fig.update_traces(marker_line_width=2.5, opacity=0.8)
    return fig
    #fig.show()


def plot_histogram_for_value_counts(df, column, color):
    fig = px.histogram(df, x=column, color_discrete_sequence=[color])
    return fig
    #fig.show()


# remove punctuations
def remove_punctuation(row):
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#',
              '*', '+', '\\',
              '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×',
              '§', '″', '′',
              '█', '…', '“', '★', '”', '–', '●', '►', '−', '¢', '¬', '░', '¡', '¶', '↑', '±', '¿', '▾', '═', '¦', '║',
              '―', '¥', '▓',
              '—', '‹', '─', '▒', '：', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', '¯', '♦', '¤', '▲', '¸',
              '⋅', '‘', '∞',
              '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '・', '╦', '╣', '╔', '╗', '▬', '❤', '≤', '‡', '√',
              '◄', '━',
              '⇒', '▶', '≥', '╝', '♡', '◊', '。', '✈', '≡', '☺', '✔', '↵', '≈', '✓', '♣', '☎', '℃', '◦', '└', '‟', '～',
              '！', '○',
              '◆', '№', '♠', '▌', '✿', '▸', '⁄', '□', '❖', '✦', '．', '÷', '｜', '┃', '／', '￥', '╠', '↩', '✭', '▐', '☼',
              '☻', '┐',
              '├', '«', '∼', '┌', '℉', '☮', '฿', '≦', '♬', '✧', '〉', '－', '⌂', '✖', '･', '◕', '※', '‖', '◀', '‰',
              '\x97', '↺',
              '∆', '┘', '┬', '╬', '،', '⌘', '⊂', '＞', '〈', '⎙', '？', '☠', '⇐', '▫', '∗', '∈', '≠', '♀', '♔', '˚', '℗',
              '┗', '＊',
              '┼', '❀', '＆', '∩', '♂', '‿', '∑', '‣', '➜', '┛', '⇓', '☯', '⊖', '☀', '┳', '；', '∇', '⇑', '✰', '◇', '♯',
              '☞', '´',
              '↔', '┏', '｡', '◘', '∂', '✌', '♭', '┣', '┴', '┓', '✨', '\xa0', '˜', '❥', '┫', '℠', '✒', '［', '∫', '\x93',
              '≧', '］',
              '\x94', '∀', '♛', '\x96', '∨', '◎', '↻', '⇩', '＜', '≫', '✩', '✪', '♕', '؟', '₤', '☛', '╮', '␊', '＋', '┈',
              '％',
              '╋', '▽', '⇨', '┻', '⊗', '￡', '।', '▂', '✯', '▇', '＿', '➤', '✞', '＝', '▷', '△', '◙', '▅', '✝', '∧', '␉',
              '☭',
              '┊', '╯', '☾', '➔', '∴', '\x92', '▃', '↳', '＾', '׳', '➢', '╭', '➡', '＠', '⊙', '☢', '˝', '∏', '„', '∥',
              '❝', '☐',
              '▆', '╱', '⋙', '๏', '☁', '⇔', '▔', '\x91', '➚', '◡', '╰', '\x85', '♢', '˙', '۞', '✘', '✮', '☑', '⋆', 'ⓘ',
              '❒',
              '☣', '✉', '⌊', '➠', '∣', '❑', '◢', 'ⓒ', '\x80', '〒', '∕', '▮', '⦿', '✫', '✚', '⋯', '♩', '☂', '❞', '‗',
              '܂', '☜',
              '‾', '✜', '╲', '∘', '⟩', '＼', '⟨', '·', '✗', '♚', '∅', 'ⓔ', '◣', '͡', '‛', '❦', '◠', '✄', '❄', '∃', '␣',
              '≪', '｢',
              '≅', '◯', '☽', '∎', '｣', '❧', '̅', 'ⓐ', '↘', '⚓', '▣', '˘', '∪', '⇢', '✍', '⊥', '＃', '⎯', '↠', '۩', '☰',
              '◥',
              '⊆', '✽', '⚡', '↪', '❁', '☹', '◼', '☃', '◤', '❏', 'ⓢ', '⊱', '➝', '̣', '✡', '∠', '｀', '▴', '┤', '∝', '♏',
              'ⓐ',
              '✎', ';', '␤', '＇', '❣', '✂', '✤', 'ⓞ', '☪', '✴', '⌒', '˛', '♒', '＄', '✶', '▻', 'ⓔ', '◌', '◈', '❚', '❂',
              '￦',
              '◉', '╜', '̃', '✱', '╖', '❉', 'ⓡ', '↗', 'ⓣ', '♻', '➽', '׀', '✲', '✬', '☉', '▉', '≒', '☥', '⌐', '♨', '✕',
              'ⓝ',
              '⊰', '❘', '＂', '⇧', '̵', '➪', '▁', '▏', '⊃', 'ⓛ', '‚', '♰', '́', '✏', '⏑', '̶', 'ⓢ', '⩾', '￠', '❍', '≃',
              '⋰', '♋',
              '､', '̂', '❋', '✳', 'ⓤ', '╤', '▕', '⌣', '✸', '℮', '⁺', '▨', '╨', 'ⓥ', '♈', '❃', '☝', '✻', '⊇', '≻', '♘',
              '♞',
              '◂', '✟', '⌠', '✠', '☚', '✥', '❊', 'ⓒ', '⌈', '❅', 'ⓡ', '♧', 'ⓞ', '▭', '❱', 'ⓣ', '∟', '☕', '♺', '∵', '⍝',
              'ⓑ',
              '✵', '✣', '٭', '♆', 'ⓘ', '∶', '⚜', '◞', '்', '✹', '➥', '↕', '̳', '∷', '✋', '➧', '∋', '̿', 'ͧ', '┅', '⥤',
              '⬆', '⋱',
              '☄', '↖', '⋮', '۔', '♌', 'ⓛ', '╕', '♓', '❯', '♍', '▋', '✺', '⭐', '✾', '♊', '➣', '▿', 'ⓑ', '♉', '⏠', '◾',
              '▹',
              '⩽', '↦', '╥', '⍵', '⌋', '։', '➨', '∮', '⇥', 'ⓗ', 'ⓓ', '⁻', '⎝', '⌥', '⌉', '◔', '◑', '✼', '♎', '♐', '╪',
              '⊚',
              '☒', '⇤', 'ⓜ', '⎠', '◐', '⚠', '╞', '◗', '⎕', 'ⓨ', '☟', 'ⓟ', '♟', '❈', '↬', 'ⓓ', '◻', '♮', '❙', '♤', '∉',
              '؛',
              '⁂', 'ⓝ', '־', '♑', '╫', '╓', '╳', '⬅', '☔', '☸', '┄', '╧', '׃', '⎢', '❆', '⋄', '⚫', '̏', '☏', '➞', '͂',
              '␙',
              'ⓤ', '◟', '̊', '⚐', '✙', '↙', '̾', '℘', '✷', '⍺', '❌', '⊢', '▵', '✅', 'ⓖ', '☨', '▰', '╡', 'ⓜ', '☤', '∽',
              '╘',
              '˹', '↨', '♙', '⬇', '♱', '⌡', '⠀', '╛', '❕', '┉', 'ⓟ', '̀', '♖', 'ⓚ', '┆', '⎜', '◜', '⚾', '⤴', '✇', '╟',
              '⎛',
              '☩', '➲', '➟', 'ⓥ', 'ⓗ', '⏝', '◃', '╢', '↯', '✆', '˃', '⍴', '❇', '⚽', '╒', '̸', '♜', '☓', '➳', '⇄', '☬',
              '⚑',
              '✐', '⌃', '◅', '▢', '❐', '∊', '☈', '॥', '⎮', '▩', 'ு', '⊹', '‵', '␔', '☊', '➸', '̌', '☿', '⇉', '⊳', '╙',
              'ⓦ',
              '⇣', '｛', '̄', '↝', '⎟', '▍', '❗', '״', '΄', '▞', '◁', '⛄', '⇝', '⎪', '♁', '⇠', '☇', '✊', 'ி', '｝', '⭕',
              '➘',
              '⁀', '☙', '❛', '❓', '⟲', '⇀', '≲', 'ⓕ', '⎥', '\u06dd', 'ͤ', '₋', '̱', '̎', '♝', '≳', '▙', '➭', '܀', 'ⓖ',
              '⇛', '▊',
              '⇗', '̷', '⇱', '℅', 'ⓧ', '⚛', '̐', '̕', '⇌', '␀', '≌', 'ⓦ', '⊤', '̓', '☦', 'ⓕ', '▜', '➙', 'ⓨ', '⌨', '◮',
              '☷',
              '◍', 'ⓚ', '≔', '⏩', '⍳', '℞', '┋', '˻', '▚', '≺', 'ْ', '▟', '➻', '̪', '⏪', '̉', '⎞', '┇', '⍟', '⇪', '▎',
              '⇦', '␝',
              '⤷', '≖', '⟶', '♗', '̴', '♄', 'ͨ', '̈', '❜', '̡', '▛', '✁', '➩', 'ா', '˂', '↥', '⏎', '⎷', '̲', '➖', '↲',
              '⩵', '̗', '❢',
              '≎', '⚔', '⇇', '̑', '⊿', '̖', '☍', '➹', '⥊', '⁁', '✢']
    row = str(row)
    for punct in puncts:
        if punct in row:
            row = row.replace(punct, ' ')
    return row


# correct mispelled words
def spell_correction(row):
    mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                    'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                    'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                    'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                    'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                    'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',
                    'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation',
                    'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis',
                    'Etherium': 'bitcoin', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017',
                    '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess',
                    "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                    'demonitization': 'demonetization', 'demonetisation': 'demonetization',
                    'electroneum': 'bitcoin', 'nanodegree': 'degree', 'hotstar': 'star', 'dream11': 'dream',
                    'ftre': 'fire', 'tensorflow': 'framework', 'unocoin': 'bitcoin',
                    'lnmiit': 'limit', 'unacademy': 'academy', 'altcoin': 'bitcoin', 'altcoins': 'bitcoin',
                    'litecoin': 'bitcoin', 'coinbase': 'bitcoin', 'cryptocurency': 'cryptocurrency',
                    'simpliv': 'simple', 'quoras': 'quora', 'schizoids': 'psychopath', 'remainers': 'remainder',
                    'twinflame': 'soulmate', 'quorans': 'quora', 'brexit': 'demonetized',
                    'iiest': 'institute', 'dceu': 'comics', 'pessat': 'exam', 'uceed': 'college', 'bhakts': 'devotee',
                    'boruto': 'anime',
                    'cryptocoin': 'bitcoin', 'blockchains': 'blockchain', 'fiancee': 'fiance', 'redmi': 'smartphone',
                    'oneplus': 'smartphone', 'qoura': 'quora', 'deepmind': 'framework', 'ryzen': 'cpu',
                    'whattsapp': 'whatsapp',
                    'undertale': 'adventure', 'zenfone': 'smartphone', 'cryptocurencies': 'cryptocurrencies',
                    'koinex': 'bitcoin', 'zebpay': 'bitcoin', 'binance': 'bitcoin', 'whtsapp': 'whatsapp',
                    'reactjs': 'framework', 'bittrex': 'bitcoin', 'bitconnect': 'bitcoin', 'bitfinex': 'bitcoin',
                    'yourquote': 'your quote', 'whyis': 'why is', 'jiophone': 'smartphone',
                    'dogecoin': 'bitcoin', 'onecoin': 'bitcoin', 'poloniex': 'bitcoin', '7700k': 'cpu',
                    'angular2': 'framework', 'segwit2x': 'bitcoin', 'hashflare': 'bitcoin', '940mx': 'gpu',
                    'openai': 'framework', 'hashflare': 'bitcoin', '1050ti': 'gpu', 'nearbuy': 'near buy',
                    'freebitco': 'bitcoin', 'antminer': 'bitcoin', 'filecoin': 'bitcoin', 'whatapp': 'whatsapp',
                    'empowr': 'empower', '1080ti': 'gpu', 'crytocurrency': 'cryptocurrency', '8700k': 'cpu',
                    'whatsaap': 'whatsapp', 'g4560': 'cpu', 'payymoney': 'pay money',
                    'fuckboys': 'fuck boys', 'intenship': 'internship', 'zcash': 'bitcoin',
                    'demonatisation': 'demonetization', 'narcicist': 'narcissist', 'mastuburation': 'masturbation',
                    'trignometric': 'trigonometric', 'cryptocurreny': 'cryptocurrency', 'howdid': 'how did',
                    'crytocurrencies': 'cryptocurrencies', 'phycopath': 'psychopath',
                    'bytecoin': 'bitcoin', 'possesiveness': 'possessiveness', 'scollege': 'college',
                    'humanties': 'humanities', 'altacoin': 'bitcoin', 'demonitised': 'demonetized',
                    'brasília': 'brazilia', 'accolite': 'accolyte', 'econimics': 'economics', 'varrier': 'warrier',
                    'quroa': 'quora', 'statergy': 'strategy', 'langague': 'language',
                    'splatoon': 'game', '7600k': 'cpu', 'gate2018': 'gate 2018', 'in2018': 'in 2018',
                    'narcassist': 'narcissist', 'jiocoin': 'bitcoin', 'hnlu': 'hulu', '7300hq': 'cpu',
                    'weatern': 'western', 'interledger': 'blockchain', 'deplation': 'deflation',
                    'cryptocurrencies': 'cryptocurrency', 'bitcoin': 'blockchain cryptocurrency', }
    words = row.split()
    for i in range(0, len(words)):
        if mispell_dict.get(words[i]) is not None:
            words[i] = mispell_dict.get(words[i])
        elif mispell_dict.get(words[i].lower()) is not None:
            words[i] = mispell_dict.get(words[i].lower())

    words = " ".join(words)
    return words


# clean contracted words
def clean_contractions(row):
    contraction_mapping = {"We'd": "We had", "That'd": "That had", "AREN'T": "Are not", "HADN'T": "Had not",
                           "Could've": "Could have", "LeT's": "Let us", "How'll": "How will", "They'll": "They will",
                           "DOESN'T": "Does not", "HE'S": "He has", "O'Clock": "Of the clock", "Who'll": "Who will",
                           "What'S": "What is", "Ain't": "Am not", "WEREN'T": "Were not", "Y'all": "You all",
                           "Y'ALL": "You all", "Here's": "Here is", "It'd": "It had", "Should've": "Should have",
                           "I'M": "I am", "ISN'T": "Is not", "Would've": "Would have", "He'll": "He will",
                           "DON'T": "Do not", "She'd": "She had", "WOULDN'T": "Would not", "She'll": "She will",
                           "IT's": "It is", "There'd": "There had", "It'll": "It will", "You'll": "You will",
                           "He'd": "He had", "What'll": "What will", "Ma'am": "Madam", "CAN'T": "Can not",
                           "THAT'S": "That is", "You've": "You have", "She's": "She is", "Weren't": "Were not",
                           "They've": "They have", "Couldn't": "Could not", "When's": "When is", "Haven't": "Have not",
                           "We'll": "We will", "That's": "That is", "We're": "We are", "They're": "They' are",
                           "You'd": "You would", "How'd": "How did", "What're": "What are", "Hasn't": "Has not",
                           "Wasn't": "Was not", "Won't": "Will not", "There's": "There is", "Didn't": "Did not",
                           "Doesn't": "Does not", "You're": "You are", "He's": "He is", "SO's": "So is",
                           "We've": "We have", "Who's": "Who is", "Wouldn't": "Would not", "Why's": "Why is",
                           "WHO's": "Who is", "Let's": "Let us", "How's": "How is", "Can't": "Can not",
                           "Where's": "Where is", "They'd": "They had", "Don't": "Do not", "Shouldn't": "Should not",
                           "Aren't": "Are not", "ain't": "is not", "What's": "What is", "It's": "It is",
                           "Isn't": "Is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                           "could've": "could have", "couldn't": "could not", "didn't": "did not",
                           "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                           "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
                           "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                           "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                           "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                           "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                           "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                           "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                           "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                           "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                           "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                           "she'll've": "she will have", "she's": "she is", "should've": "should have",
                           "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                           "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                           "that's": "that is", "there'd": "there would", "there'd've": "there would have",
                           "there's": "there is", "here's": "here is", "they'd": "they would",
                           "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                           "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                           "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                           "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
                           "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                           "where'd": "where did", "where's": "where is", "where've": "where have",
                           "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                           "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                           "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                           "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                           "you'll've": "you will have", "you're": "you are", "you've": "you have"}
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        row = row.replace(s, "'")

    row = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in row.split(" ")])
    return row



# remove stop words
def remove_stopwords(x):
    x = [word for word in x.split() if word not in STOPWORDS]
    x = ' '.join(x)

    return x



# lemmatization
def lemmatize(row):
    lemmatizer = WordNetLemmatizer()
    row = row.split()
    row = [lemmatizer.lemmatize(word) for word in row]
    row = ' '.join(row)

    return row


def remove_html(row):
    regex = re.compile(r'<[^>]+>')
    return regex.sub('', str(row))


def remove_single_letters(row):
    row = str(row)
    row = row.split()
    lst = []
    for word in row:
        if len(word) > 1:
            lst.append(word)

    return ' '.join(lst)


def preprocess(df):
    for essay in df.columns:
        df[essay] = df[essay].apply(lambda row : remove_html(row))
        df[essay] = df[essay].apply(lambda row : remove_punctuation(row))
        df[essay] = df[essay].apply(lambda row : clean_contractions(row))
        df[essay] = df[essay].apply(lambda row : spell_correction(row))
        df[essay] = df[essay].apply(lambda row : remove_stopwords(row))
        df[essay] = df[essay].apply(lambda row : lemmatize(row))
        df[essay] = df[essay].apply(lambda row : remove_single_letters(row))


def createCloud(text, title, size=(10, 7)):
    # Processing Text
    wordcloud = WordCloud(width=800, height=400,
                          collocations=False
                          ).generate(" ".join(text))

    # Output Visualization
    fig = plt.figure(figsize=size, dpi=80, facecolor='k', edgecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=25, color='w')
    plt.tight_layout(pad=0)
    return fig


def top_n_words(text, ngram, n):
    if (ngram == 1):
        vec = CountVectorizer().fit(text)
    else:
        vec = CountVectorizer(ngram_range=(ngram, ngram)).fit(text)

    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def plot_word_freq_chart(word_freq_list):
    x = []
    y = []

    for word, freq in word_freq_list:
        x.append(word)
        y.append(freq)

    data = [go.Bar(
        x=x,
        y=y
    )]

    fig = go.Figure(data=data)
    fig.update_layout(
        title="Top 10 Words",
        xaxis_title="Word",
        yaxis_title="Frequency",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ),
    )
    fig.update_traces(marker_color='lightgreen', marker_line_color='navy',
                      marker_line_width=2.5, opacity=0.6)
    #iplot(fig)
    return fig


def gender_vs_income(df):
    male_income = df[df["sex"] == "m"]["income"].mean()
    female_income = df[df["sex"] == "f"]["income"].mean()
    data = [go.Bar(
        x=["Male", "Female"],
        y=[male_income, female_income]
    )]

    fig = go.Figure(data=data)

    fig.update_layout(
        xaxis_title="Gender",
        yaxis_title="Income(in USD)",
        font=dict(
            family="Courier New, monospace",
            size=18
        ),
    )

    return fig


def status_vs_income(df):
    status = []
    income = []
    for s in df.status.value_counts().index:
        status.append(s)
        income.append(df[df["status"] == s]["income"].mean())

    data = [go.Bar(
        x=status,
        y=income,
        marker_color="#e377c2"
    )]

    fig = go.Figure(data=data)

    fig.update_layout(
        xaxis_title="Gender",
        yaxis_title="Income(in USD)",
        font=dict(
            family="Courier New, monospace",
            size=18
        ),
    )
    return fig


def offspring_vs_income(df):
    df["offspring"] = df["offspring"].apply(lambda row: str(row))
    df["offspring"] = df["offspring"].apply(lambda row: row.replace("&rsquo;", "'"))

    offspring = []
    income = []
    for s in df.offspring.value_counts().index:
        offspring.append(s)
        income.append(df[df["offspring"] == s]["income"].mean())

    offspring[0] = "Not Defined"
    data = [go.Bar(
        x=income,
        y=offspring,
        marker_color="#ff7f0e",
        orientation='h'
    )]

    fig = go.Figure(data=data)

    fig.update_layout(
        xaxis_title="Income",
        yaxis_title="Offsprings"
    )

    return fig


def body_type_vs_relationship_status(df, body):
    labels = ["single", "other"]

    singles = df[df["body_type"]==body]["status"].value_counts()["single"]
    total = df[df["body_type"]==body]["status"].value_counts().sum()
    fig = go.Figure(data=[go.Pie(labels=labels, values=[singles, total-singles], hole=.6)])
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                      annotations=[dict(text=body, x=0.5, y=0.5, font_size=11, showarrow=False)])
    fig.update_traces(hoverinfo='label + percent + value', textinfo = "none")
    return fig


def body_type_vs_gender(df, body):
    labels = ["Male", "Female"]

    males = df[df["body_type"]==body]["sex"].value_counts()["m"]
    females = df[df["body_type"]==body]["sex"].value_counts()["f"]

    fig = go.Figure(data=[go.Pie(labels=labels, values=[males, females], hole=.6)])
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0),
                      annotations=[dict(text=body, x=0.5, y=0.5, font_size=11, showarrow=False)])
    fig.update_traces(hoverinfo='label + percent + value', textinfo='none', marker=dict(colors=["darkblue", "cyan"]))

    return fig


def app():
    st.header("Love is in the Air!")
    li = []

    for filename in ["profiles - profiles.csv", "profiles1 - profiles1.csv", "profiles2 - profiles2.csv", "profiles3 - profiles3.csv", "profiles4 - profiles4.csv", "profiles5 - profiles5.csv", "profiles6 - profiles6.csv", "profiles7 - profiles7.csv"]:
        df = pd.read_csv(filename, index_col=None, header=None)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    new_header = df.iloc[0] #grab the first row for the header
    df = df[1:] #take the data less the header row
    df.columns = new_header #set the header row as the df header
    st.subheader("OkCupid Dataset")
    st.write(df.head(10))

    st.header("Understanding the Dataset")
    df['income'] = df['income'].apply(lambda row : float(row))
    df['age'] = df['age'].apply(lambda row : float(row))
    st.subheader("Age Distribution")
    st.plotly_chart(plot_histogram_for_value_counts(df, "age", "lightblue"))

    st.subheader("Body Types Distribution")
    st.plotly_chart(plot_histogram_for_value_counts(df, "body_type", "indianred"))

    st.subheader("Diet Distribution")
    st.plotly_chart(plot_histogram_for_value_counts(df, "diet", "lightgreen"))

    col1, col2 = st.columns(2)
    col1.subheader("Gender Distribution")
    col1.plotly_chart(plot_pie_chart_for_value_counts(df.sex), use_container_width = True)
    col2.subheader("Smoking Habits")
    col2.plotly_chart(plot_pie_chart_for_value_counts(df.smokes), use_container_width = True)

    col1, col2 = st.columns(2)
    col1.subheader("Drugs Habits")
    col1.plotly_chart(plot_pie_chart_for_value_counts(df.drugs), use_container_width = True)
    col2.subheader("Smoking Habits")
    col2.plotly_chart(plot_pie_chart_for_value_counts(df.drinks), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.subheader("Relationship Status")
    col1.plotly_chart(plot_pie_chart_for_value_counts(df.status), use_container_width=True)
    col2.subheader("Orientation")
    col2.plotly_chart(plot_pie_chart_for_value_counts(df.orientation), use_container_width=True)



    st.subheader("Gender Vs Income")
    st.plotly_chart(gender_vs_income(df))

    st.subheader("Relationship Status Vs Income")
    st.plotly_chart(status_vs_income(df))

    st.subheader("Offsprings Vs Income")
    st.plotly_chart(offspring_vs_income(df))

    st.subheader("Body Type vs Status")
    col1, col2, col3 = st.columns(3)
    #col1.subheader("Fit")
    col1.plotly_chart(body_type_vs_relationship_status(df, "fit"), use_container_width=True)
    #col2.subheader("Average")
    col2.plotly_chart(body_type_vs_relationship_status(df, "jacked"), use_container_width=True)
    #col3.subheader("A Little Extra")
    col3.plotly_chart(body_type_vs_relationship_status(df, "curvy"), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    # col1.subheader("Fit")
    col1.plotly_chart(body_type_vs_relationship_status(df, "average"), use_container_width=True)
    # col2.subheader("Average")
    col2.plotly_chart(body_type_vs_relationship_status(df, "skinny"), use_container_width=True)
    # col3.subheader("A Little Extra")
    col3.plotly_chart(body_type_vs_relationship_status(df, "thin"), use_container_width=True)
    
    
    st.subheader("Body Type vs Gender")
    col1, col2, col3 = st.columns(3)
    # col1.subheader("Fit")
    col1.plotly_chart(body_type_vs_gender(df, "fit"), use_container_width=True)
    # col2.subheader("Average")
    col2.plotly_chart(body_type_vs_gender(df, "jacked"), use_container_width=True)
    # col3.subheader("A Little Extra")
    col3.plotly_chart(body_type_vs_gender(df, "curvy"), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    # col1.subheader("Fit")
    col1.plotly_chart(body_type_vs_gender(df, "average"), use_container_width=True)
    # col2.subheader("Average")
    col2.plotly_chart(body_type_vs_gender(df, "skinny"), use_container_width=True)
    # col3.subheader("A Little Extra")
    col3.plotly_chart(body_type_vs_gender(df, "thin"), use_container_width=True)

    st.header("Visualize the Essays")
    essays_df = pd.read_csv("processed-essays.csv")
    #preprocess(essays_df)
    
    for essay in essays_df.columns:
        essays_df[essay] = essays_df[essay].apply(lambda row : str(row))
        essays_df[essay] = essays_df[essay].apply(lambda row : row.replace("nan", ""))


    st.write(essays_df.head(10))

    st.subheader("Choose The Essay to Visualize:")

    essay = st.selectbox(
        "Choose the essay:",
        ("essay0 - My self summary", "essay1 - What I’m doing with my life", "essay2 - I’m really good at", 
         "essay3 - The first thing people usually notice about me", "essay4 - Favorite books, movies, show, music, and food", 
         "essay5 - The six things I could never do without", "essay6 - I spend a lot of time thinking about", 
         "essay7 - On a typical Friday night I am", "essay8 - The most private thing I am willing to admit", 
         "essay9 - You should message me if…"))

    st.subheader("Word Cloud")
    st.pyplot(createCloud(essays_df[essay[:6]], essay))


    st.header("Bigrams")
    st.plotly_chart(plot_word_freq_chart(top_n_words(essays_df[essay[:6]], 2, 10)), use_container_width = True)

    st.header("Trigrams")
    st.plotly_chart(plot_word_freq_chart(top_n_words(essays_df[essay[:6]], 3, 10)), use_container_width = True)


app()
