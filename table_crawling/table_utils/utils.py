from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import urllib.parse
import re
import urllib3
# HTTP manager
http = urllib3.PoolManager()
urllib3.disable_warnings()

def url2dockey(string):
    string = urllib.parse.unquote(string)
    return string

def tokenize(string, remove_dot=False):
    def func(string):
        return " ".join(word_tokenize(string))
    
    string = string.replace('%-', '-')
    if remove_dot:
        string = string.rstrip('.')

    string = func(string)
    string = string.replace(' %', '%')
    return string

def normalize(string):
    string = string.strip().replace('\n', ' ')
    return tokenize(string)

def filter_firstKsents(string, k):
    string = sent_tokenize(string)
    string = string[:k]
    return " ".join(string)

def process_link(text):
    tmp = []
    hrefs = []
    for t in text.find_all('a'):
        if len(t.get_text().strip()) > 0:
            if 'href' in t.attrs and t['href'].startswith('/wiki/'):
                tmp.append(t.get_text(separator=" ").strip())
                hrefs.append(t['href'].split('#')[0])

    return hrefs

def remove_ref(text):
    for x in text.find_all('sup'):
        x.extract()
    return text

def clean_cell_text(string):
    string = string.replace('"', '')
    string = string.rstrip('^')
    string = string.replace('–', '-')
    #string = re.sub(r'\b([0-9]{4})-', r'\1 - ', string)    
    #string = re.sub(r'^([0-9]{1,2})-([0-9]{1,2})$', r'\1 - \2', string)
    #string = re.sub(r'^([0-9]{1,2})-([0-9]{1,2})-([0-9]{1,2})$', r'\1 - \2 - \3', string)
    string = string.replace('( ', '(')
    string = string.replace(' )', ')')
    string = string.replace('"', '')
    string = string.replace(u"\u00a0", u' ')
    string = string.replace('\n', ' ')
    string = string.rstrip('^')
    string = string.replace('\u200e', '')
    string = string.replace('\ufeff', '')
    string = string.replace(u'\u2009', u' ')
    string = string.replace(u'\u2010', u' - ')
    string = string.replace(u'\u2011', u' - ')
    string = string.replace(u'\u2012', u' - ')
    string = string.replace(u'\u2013', u' - ')
    string = string.replace(u'\u2014', u' - ')
    string = string.replace(u'\u2015', u' - ')
    string = string.replace(u'\u2018', u'')
    string = string.replace(u'\u2019', u'')
    string = string.replace(u'\u201c', u'')
    string = string.replace(u'\u201d', u'')
    string = re.sub(r' +', ' ', string)
    string = string.strip()
    return string

def clean_text(string):
    position = string.find("mw-parser-output")
    if position != -1:
        left_quote = position - 1
        while left_quote >= 0 and string[left_quote] != '(':
            left_quote -= 1
        right_quote = position + 1
        while right_quote < len(string) and string[right_quote] != ')':
            right_quote += 1
        
        string = string[:left_quote] + " " + string[right_quote + 1:]
        
        position = string.find("mw-parser-output")
        if position != -1:
            #print(string)
            right_quote = position + 1
            while right_quote < len(string) and string[right_quote] != '\n':
                right_quote += 1
            #print("----------------")
            string = string[:position] + string[right_quote + 1:]
            #print(string)
            #print("################")
    
    string = re.sub(r'\[[\d]+\]', '', string).strip()
    string = string.replace(u'\xa0', u' ')
    string = string.replace('\ufeff', '')
    string = string.replace(u'\u200e', u' ')
    string = string.replace('–', '-')
    string = string.replace(u'\u2009', u' ')
    string = string.replace(u'\u2010', u' - ')
    string = string.replace(u'\u2011', u' - ')
    string = string.replace(u'\u2012', u' - ')
    string = string.replace(u'\u2013', u' - ')
    string = string.replace(u'\u2014', u' - ')
    string = string.replace(u'\u2015', u' - ')
    string = string.replace(u'\u2018', u'')
    string = string.replace(u'\u2019', u'')
    string = string.replace(u'\u201c', u'')
    string = string.replace(u'\u201d', u'')    
    
    string = string.replace(u'"', u'')
    string = re.sub(r'[\n]+', '\n', string)
    
    string = re.sub(r'\.+', '.', string)
    string = re.sub(r' +', ' ', string)
    
    #string = re.sub(r"'+", "'", string)
    #string = string.replace(" '", " ")
    #string = string.replace("' ", " ")
    string = filter_firstKsents(string, 12)
    
    return string

def get_section_title(r):
    text = r.previous_sibling
    title_hierarchy = []
    while text is None or text == '\n' or text.name not in ['h2', 'h3']:
        if text is None:
            break
        else:
            text = text.previous_sibling               
    
    if text is not None:
        title_hierarchy.append(text.find(class_='mw-headline').text)
        if text.name in ['h3']:
            while text is None or text == '\n' or text.name not in ['h2']:
                if text is None:
                    break
                else:
                    text = text.previous_sibling               

            if text is None:
                pass
            else:
                title_hierarchy.append(text.find(class_='mw-headline').text)
    
    if len(title_hierarchy) == 0:
        return ''
    else:
        tmp = ' -- '.join(title_hierarchy[::-1])
        return normalize(tmp)

def get_section_text(r):
    text = r.previous_sibling
    section_text = ''
    while text is not None:
        if text == '\n':
            text = text.previous_sibling
        elif text.name in ['h1', 'h2', 'h3', 'h4']:
            break
        else:
            tmp = text.text
            if tmp:
                mask = ['note', 'indicate', 'incomplete', 'source', 'reference']
                if  any([_ in tmp.lower() for _ in mask]):
                    tmp = ''
                else:
                    tmp = normalize(tmp)
                    if section_text:
                        section_text = tmp + ' ' + section_text
                    else:
                        section_text = tmp
            text = text.previous_sibling
    return section_text


def download_summary(page):
    if page.startswith('https'):
        pass
    elif page.startswith('/wiki'):
        page = 'https://en.wikipedia.org{}'.format(page)
    else:
        page = 'https://en.wikipedia.org/wiki/{}'.format(page)
    
    r = http.request('GET', page)
    if r.status == 200:
        data = r.data.decode('utf-8')
        data = data.replace('</p><p>', ' ') 
        soup = BeautifulSoup(data, 'html.parser')
        div = soup.body.find("div", {"class": "mw-parser-output"})
        if div:
            children = div.findChildren("p" , recursive=False)
            summary = 'N/A'
            for child in children:
                if child.get_text().strip() != "":
                    html = str(child)
                    html = html[html.index('>') + 1:].strip()
                    if not html.startswith('<'):
                        summary = child.get_text(separator=" ").strip()
                        break
                    elif html.startswith('<a>') or html.startswith('<b>') or \
                            html.startswith('<i>') or html.startswith('<a ') or html.startswith('<br>'):
                        summary = child.get_text(separator=" ").strip()
                        break
                    else:
                        continue
            return summary
        else:
            return 'N/A'
    elif r.status == 429:
        time.sleep(1)
        return download_summary(page)
    elif r.status == 404:
        return 'N/A'
    else:
        raise ValueError("return with code {}".format(r.status))