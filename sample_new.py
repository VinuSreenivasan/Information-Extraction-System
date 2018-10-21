import sys
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from stanfordcorenlp import StanfordCoreNLP
from re import search
import re
import string
import os
from nltk.tag import StanfordNERTagger
import spacy
#from spacy.en import English
from fuzzywuzzy import fuzz
#from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
ennlp=spacy.load('en')

wd=WordNetLemmatizer()
stanford_classifier = os.environ.get('STANFORD_MODELS').split(":")[0]
stanford_ner_path = os.environ.get('CLASSPATH').split(":")[0]
st=StanfordNERTagger(stanford_classifier, stanford_ner_path, encoding='utf-8')

stop = stopwords.words('english')
nlp = StanfordCoreNLP(r'stanford-corenlp-full-2017-06-09/')

template = sys.argv[1]+".templates"
fout = open(template,"w")

ALPHA = 0.4
BETA = 0.6

incident_list = ["arson","attack","bombing","kidnapping","robbery"]
weapon_list = []
target_list = []
skip_list = []

list_of_keywords=[]
with open("individual.txt") as perp:
	for keyword in perp:
		list_of_keywords.append(keyword.rstrip("\r\n"))

def create_victim():
	victims = []
	with open("victim.txt","r") as f1: 
		for i in f1:
			i=i.rstrip()
			victims.append(i)

	victims = list(set(victims))
	return victims


#GENERAL INDIVIDUAL
def general():
	general_indiv=[]
	with open("names.txt","r") as f1:
		for i in f1:
			general_indiv.append(i.strip())
	#print skipwords
	return general_indiv

def getFeatures(answerfolder,textfolder):
	example=[]
	label=[]
	for files in os.listdir(textfolder):
		ansextension="".join([files,'.anskey'])
		textpath=os.path.join(textfolder,files)
		anspath=os.path.join(answerfolder,ansextension)
		ftext=open(textpath)
		flabel=open(anspath)
		#ftext.readline()
		text=ftext.read()
		example.append(text)
		answer=flabel.read()
		answer=answer.split('\n')
		for slots in answer:
			slot=slots.split(':')
			if slot[0] == 'INCIDENT':
				l=slot[-1].strip()
				label.append(l)
		ftext.close()
		flabel.close()
	return example,label	

def test_file(filename):
	corpus=[]
	corpus1=[]
	document=[]
	header=[]
	with open(filename,'r') as f1:
		p=re.compile(r'(?:D|T)(?:E|S)(?:V|T)[0-9]*\-MUC[0-9]\-[0-9]{4}')
		for sent in f1.readlines():
			sent1=sent.split('\n')
			p1=p.match(sent1[0])
			if p.match(sent1[0]) is None:
				document.append(sent)
			else:
				corpus.append("".join(document))
				header.append(p1.group())
				document=[sent]
	#print len(header)
	corpus.append("".join(document))
	#print len(corpus[1:])	
	return header,corpus

#INCIDENT MAIN	
def incidents():
	incident_dict={}
	answerfolder='developset/answers'
	textfolder='developset/texts'
	featureRaw= getFeatures(answerfolder,textfolder)
	x_train,y_train=featureRaw[0],featureRaw[-1]
	countVec=TfidfVectorizer(stop_words='english')
	x_trainCV=countVec.fit_transform(x_train)
	T=x_trainCV.toarray()
	id_values, corpus=test_file(sys.argv[1])
	x_testCV=countVec.transform(corpus[1:])
	L=x_testCV.toarray()
	svmtest=LinearSVC().fit(T, y_train)
	svm=svmtest.predict(L)
	incident_dict = dict(zip(id_values, svm))
	#print id_values
	#print incident_dict
	return incident_dict


def incident_value(idvalue):
	final_incident = incident_dict[idvalue]
	return final_incident

#CHECKING FOR POSITIONS OF PREP
def posof(sent):
	pos=-1
	for i in range(len(sent)):
		#print sent[i]
		if sent[i] in 'of':
			pos=i
			break
		elif sent[i] in 'against':
			pos=i
			break
	if pos!=-1:
		return pos
	else:
		return -1

#REMOVE DET AND COUNTS
def check(np):
	sp=np.split()
	first=nlp.pos_tag(sp[0])
	#print first[0][1]		
	if(first[0][1] == "DT" or first[0][1]=="CD"):
		l=np.lstrip(first[0][0])
		return str(l)
	else:
		return np
	
#VICTIM
def victim(paragraph):
	newlist=[]
	humannames2=namesfunc(paragraph)    #extraction of human names from the para
	humannames1=set(humannames2)
	for i in humannames1: 
		for j in humannames1:
			if i in j and i!=j:
				newlist.append(i)
	humannames = humannames1 - set(newlist)    #finding distinct human names
	#print humannames
	verblist=['VB','VBD','VBN','VBP','VBG','VBZ','JJ']
	names1=['NNS','NNP','NN']
	names2=['NNS','NNP','NN']
	impwords=[]; passivesent=[]
	shortlistedsent=[]; mainsent=[]; mainsent1=[]; mainsent2=[]; mainsentnn4=[]

	sent=sent_tokenize(paragraph)
	for s in sent:
		wordset=[];
		wordtokens=word_tokenize(s)
		wordstok=filter(str.isalnum,wordtokens)
		maximum=0
		for w in wordstok:
			postags=nlp.pos_tag(w)
			stemmed=w
			stemmed1=w
			if(str(postags[0][1]) in verblist):	
				stemmed=wd.lemmatize(w,'v')
			if stemmed in victimlist:
				shortlistedsent.append(s)
				impwords.append(stemmed)
				break
	for sw in sent:
		words=word_tokenize(sw)
		gram= r'''
		OF: { "of" }
		NNN1: {<DT>*<NN|NN.><IN><DT>*<JJ>*<CD>*<NN|NN.>*<JJ>*<NN|NN.>+}
		NNN2: {<CD>* <NN|NN.>+<CC>*<NN|NN.>*<WP>*<VB.><RB|RB.><VB.>+}
		NNN3: {<VB.>+<DT>*<CD>*<JJ>*<NN|NN.>+<,>*<CC>*<DT>*<NN|NN.>*}
		'''
		chunked_text = nltk.RegexpParser(gram)
		tokenised_words=word_tokenize(sw)
		poswords = nlp.pos_tag(sw)
		a=[]
		tree = chunked_text.parse(poswords)
		for subtree in tree.subtrees(filter = lambda t: t.label()=='OF'):
			passivesent.append(" ".join([a for (a,b) in subtree.leaves()]))
		for subtree in tree.subtrees(filter = lambda t: t.label()=='NNN1'):
			mainsent.append(" ".join([a for (a,b) in subtree.leaves()]))
		for subtree in tree.subtrees(filter = lambda t: t.label()=='NNN3'):
			mainsent2.append(" ".join([a for (a,b) in subtree.leaves()]))
		for subtree in tree.subtrees(filter = lambda t: t.label()=='NNN2'):
			mainsentnn4.append(" ".join([a for (a,b) in subtree.leaves()]))
			#mainsent2.append(subtree.leaves())
	a=[]; shortlistsent=[];  ss1=" "
	#VERB NOUN-PHRASE grmmar rule
	if(mainsent2!=0):
		nounphrases1=[]; nounphrases=[];
		for s in mainsent2:
			doc = ennlp(s)
			for np in doc.noun_chunks:
				nounphrases1.append(np)
		nounphrases2 = list(set(nounphrases1))
		for i in nounphrases2:
			if i:
				nounphrases.append(i)
		#print nounphrases
		for i in mainsent2:
			tw=word_tokenize(i)
			stw=wd.lemmatize(tw[0],'v')
			#print stw
			if stw in impwords:
				shortlistsent.append(i)	
		for i in shortlistsent:
			ss1=" "
			for j in humannames:
				if j.lower() in str(i):
					ss1=j.lower()
					a.append(ss1)
			if ss1==" ":			
				for np in nounphrases:
					if str(np) in i:
						finale=check(str(np))
						if finale in general_indiv:
							a.append(finale)	
	shortsent=[]; b=[]; ss=" "
	if(mainsent!=0):
		nounphrases1=[]; nounphrases=[];
		for s in mainsent:
			#noun phrase extraction starts here
			doc = ennlp(s)
			for np in doc.noun_chunks:
				nounphrases1.append(np)
		nounphrases2 = list(set(nounphrases1))
		for i in nounphrases2:
			if i:
				nounphrases.append(i)
			#noun phrase extraction ends here
		#print nounphrases

		for i in mainsent:
			wt=word_tokenize(i)
			for w in wt:
				stemm=wd.lemmatize(w,'v')
				if stemm in victimlist:
					shortsent.append(i)
					break
		#print shortsent
		for i in shortsent:
			if "of" in i or 'against' in i or "at" in i:
			#print i
				ss=" "
				for j in humannames:
					#print i
					if j.lower() in str(i):
						ss=j
						b.append(ss)
						break
				if ss==" ":
					eachsent = str(i).split()
					#print eachsent
					pos=posof(eachsent)
					#print pos
					if(pos!= -1):
						postagging=nlp.pos_tag(eachsent[pos+1])
						if postagging[0][1] not in ['CD','DT']:	
							for k in general_indiv:
								if k in " ".join(eachsent[pos+1:]):
									#print k
									b.append(" ".join(eachsent[pos+1:]))
						else:
							postagging=nlp.pos_tag(eachsent[pos+2])
							if postagging[0][1] not in ['CD','DT']:
								for k in general_indiv:
									if k in " ".join(eachsent[pos+2:]):
										b.append(" ".join(eachsent[pos+2:]))
							else:
								for k in general_indiv:
									if k in " ".join(eachsent[pos+3:]):
										#print k
										b.append(" ".join(eachsent[pos+3:]))	
							
	#passive voice grammar
	shortsentnn4=[]; c=[]; ssnn4=" "
	if(mainsentnn4!=0):
		for i in mainsentnn4:
			wt=word_tokenize(i)
			stemm=wd.lemmatize(wt[-1],'v')
			#print stemm
			if stemm in victimlist:
				#print stemm
				shortsentnn4.append(i)
		
		#print shortsentnn4
		for i in shortsentnn4:
			ss2=" "
			for j in humannames:
				#print i
				if j.lower() in str(i):
					#print j.lower()
					ss2=j
					c.append(ss2)
					break
			if ss2 == " ":
				doc=ennlp(unicode(i))
				#for tok in doc:
				#print tok.dep_
				passive_toks=[tok for tok in doc  if (tok.dep_ == "nsubjpass")]
				if passive_toks != []:
					for p in passive_toks:
						if str(p) in general_indiv:
							c.append(str(p))
	#print c
	total=[]
	for i in a:
		total.append(i)
	for i in b:
		total.append(i)
	for i in c:
		total.append(i)	
	#print total					
	#print list(set(total))
	if total!=[]:
		total=list(set(total))
		return list(set(total))
	else:
		return -1	

def extract_id(file_name):
	f1 = open(file_name)
	unique_id=[]
	corpus=[]
	for line in f1:
		temp = line.rstrip("\n")
		corpus.append(temp)

		match = search(r"DEV-MUC3-[0-9]{4}",line) or search("TST1-MUC3-[0-9]{4}",line) or search("TST2-MUC4-[0-9]{4}",line)
		
		if match:
			unique_id.append(match.group())
	
	return unique_id, corpus


def create_weapons():
	with open("weapon.txt") as fs:
		weapons = []

		for line in fs:
			word = line.strip()
			if word not in weapons:
				weapons.append(word)

	weapons = list(set(weapons))
	return weapons


#ORGANIZATION LIST
def organisation_names(f1):
	orgnamelist=[] 
	with open(f1) as f2:
		for line in f2:
			orgname = line.strip()
			orgnamelist.append(orgname)
	final_org_name = list(set(orgnamelist))
	#print final_org_name
	return final_org_name



#ORGANIZATION NER
def orgextract(para):
	sent=sent_tokenize(para)
	nounn = ['NN','NNPS','NNP','NNS','JJ']
	list2=" "
	for i in sent:
		w=word_tokenize(i)
		p=nltk.pos_tag(w)
		list1=" "
		for k,v in p:
			if v in nounn:
				k1=string.capwords(k)
				list1=list1 + " " + k1
			else:
				list1=list1 + " " + k
		list2=list2+list1
	s=st.tag(list2.split())
	list_org=[]
	named_entities = get_continuous_chunks(s)
	named_entities_str = [" ".join([token for token, tag in ne]) for ne in named_entities]
	named_entities_str_tag = [(" ".join([token for token, tag in ne]), ne[0][1]) for ne in named_entities]
	for k,v in named_entities_str_tag:
		if v=="ORGANIZATION":
			list_org.append(k)
	return list_org
	
#ORGANIZATION SLOT
def organisation(paragraph):
	finalorg= []; nounphrases1=[]; shortlistedsent=[]
	key=['claim','members of','front','movement','actions against','responsible for']
	org_names=orgextract(paragraph)
	sent=sent_tokenize(paragraph)
	org_value="-"
	for s in sent:
		for k in key:
			if k in s:
				for f in final_org_name:
					if f in s:
						org_value = f
	return org_value


def find_pos(w, noun_splits):
	temp=0
	for i in range(len(noun_splits)):
		if noun_splits[i] == w:
			temp=i
	return temp

#WEAPON EXTRACTION
def weapons(para):
	nounphrases1=[]; finalsetweapons=[]
	prevtags=["JJ","NN","NNS","NNP","VB","VBD","VBN","VBP","VBZ","VBG",","]
	noun1=['NN','JJ','NNP','NNS']
	sent=sent_tokenize(para)
	#print type(sent)
	tag=[]
	count=0
	weapon_final_ans="-"
	ans_list = []
	for s in sent:
		doc = ennlp(unicode(s))
		for np in doc.noun_chunks:
			nounphrases1.append(np)
		#print nounphrases1
		words=word_tokenize(str(s))
		#stopped_words = [i for i in words if not i in stopwords]	
		#print pos_t			
		for w in words:
			for wp in nounphrases1:
				if w in weapon_list and w in str(wp):
					noun_chunk=str(wp)
					noun_splits=noun_chunk.split()
					pos = find_pos(w,noun_splits)
					#print pos
					if(pos!=0):
						postag = nlp.pos_tag(noun_splits[pos-1])
						if postag[0][1] == 'JJ':
							weapon_final_ans = noun_splits[pos-1] + " " + w
							ans_list.append(weapon_final_ans)
							ans_list.append(w)
						else:
							weapon_final_ans = w
							ans_list.append(weapon_final_ans)
					else:
						weapon_final_ans = w
						ans_list.append(weapon_final_ans)
						
	ans_list = list(set(ans_list))
	if ans_list:
		return ans_list
	else:
		return -1


def get_continuous_chunks(tagged_sent):
    continuous_chunk = []
    current_chunk = []

    for token, tag in tagged_sent:
        if tag != "O":
            current_chunk.append((token, tag))
        else:
            if current_chunk: # if the current chunk is not empty
                continuous_chunk.append(current_chunk)
                current_chunk = []
    # Flush the final current_chunk into the continuous_chunk, if any.
    if current_chunk:
        continuous_chunk.append(current_chunk)
    return continuous_chunk


def namesfunc(paragraph):
	s1=paragraph.lower()
	s=sent_tokenize(s1)
	nounn = ['NN','NNPS','NNP','NNS','JJ']
	list2=" "
	for i in s:
		w=word_tokenize(i)
		p=nltk.pos_tag(w)
		list1=" "
		for k,v in p:
			if v in nounn:
				k1=string.capwords(k)
				list1=list1 + " " + k1
			else:
				list1=list1 + " " + k
		list2=list2+list1
	s=st.tag(list2.split())
	list_person=[]
	named_entities = get_continuous_chunks(s)
	named_entities_str = [" ".join([token for token, tag in ne]) for ne in named_entities]
	named_entities_str_tag = [(" ".join([token for token, tag in ne]), ne[0][1]) for ne in named_entities]

	for k,v in named_entities_str_tag:
		if v=="PERSON":
			list_person.append(str(k))
	#print list_person
	return list_person

def create_target():

	target = []
	with open("target.txt","r") as f1: 
		for i in f1:
			i = i.rstrip()
			target.append(i)

	target = list(set(target))

	return target

def create_skip_indv():

	skip = []
	with open("skip_words1.txt","r") as f1: 
		for i in f1:
			i = i.rstrip()
			skip.append(i)

	skip = list(set(skip))

	return skip

def create_skip():

	skip = []
	with open("skip_words2.txt","r") as f1: 
		for i in f1:
			i = i.rstrip()
			skip.append(i)

	skip = list(set(skip))

	return skip

def target(article):

	sent_extract = []
	sent = sent_tokenize(article)

	required = ['NN1','NN2','NN3','VB1','VB2','VB3','VB4','SUB','NN4']

	gram = []
	grammer1 = r'''
	NN1: {<VB.|VB|NN.|NN>+<DT>*<IN>*<DT>*<NN|NN.>+}
	'''
	grammer2 = r'''
	NN2: {<NNS>+<IN><JJ>*<NN|NNS>*<CC>*<JJ>*<NN|NNS>+}
	'''
	grammer3 = r'''
	NN3: {<NN>*<IN>*<TO>*<CD>*<NN|NN.>+}
	'''
	grammer4 = r'''
	NN4: {<DT>*<JJ>*<NN>+<IN>+<DT>*<NN>+<VB.>*<NN>+}
	'''
	grammer5 = r'''
	VB1: {<VBG><NNS>}
	'''
	grammer6 = r'''
	VB2: {<NN>*<VB.><IN>*<DT><NN>+}
	'''
	grammer7 = r'''
	VB3: {<CD>*<NNS><VB.><IN>*<DT>*<NN>+}
	'''
	grammer8 = r'''
	VB4: {<NN>*<VBG><IN><NN><TO><NN><VB.>+}
	'''

	gram.append(grammer1)
	gram.append(grammer2)
	gram.append(grammer3)
	gram.append(grammer4)
	gram.append(grammer5)
	gram.append(grammer6)
	gram.append(grammer7)
	gram.append(grammer8)

	for ws in sent:
		words = word_tokenize(ws)
		
		for g in gram:
			after_chunk = nltk.RegexpParser(g)
			pos_words = nlp.pos_tag(ws)
			#print pos_words
			tree = after_chunk.parse(pos_words)

			for subtree in tree.subtrees(filter = lambda t: t.label() in required):
				sent_extract.append(" ".join([a for (a,b) in subtree.leaves()]))
		


	shortsent = []
	target_words = []
	if(len(sent_extract) != 0):
		#print sent_extract
		for i in sent_extract:
			wt = word_tokenize(i)

			for w in wt:
				stemm = wd.lemmatize(w,'v')

				if stemm in target_list:
					shortsent.append(i)
					target_words.append(w)
					break
		
		#print "-------------------"
		#print shortsent
		#print target_words
		ss = []
		prev = " "
		select = " "

		tags = ['JJ','NN','NNS']
		
		start = end = 0	

		for i in shortsent:
			wws = word_tokenize(i)

			for w in range(len(wws)):

				if end >= w:
					continue

				if wws[w] in target_words: # skip if its a key word
					continue
				elif wws[w] in target_list: #skip some weapons
					continue
				elif wws[w] in skip_list: #skip locations and unnecessary adjectives
					continue
				
				cur = nlp.pos_tag(wws[w])
				cur_tag = cur[0][1]

				if cur_tag in tags:
					start = w
					tmp = w

					while (tmp < len(wws)) and ((nlp.pos_tag(wws[tmp]))[0][1] in ['NNS','NN']) and (wws[tmp] not in skip_list) and (wws[tmp] not in target_list) and (wws[tmp] not in target_words):
						tmp = tmp + 1

					end = tmp 
				
					if start == end:
						ss.append(wws[start])
					else:
						ss.append(" ".join(wws[i] for i in range(start, end)))
							
						
					#break


		if (len(ss) != 0):
			ss = list(set(ss))
			return ss
		else:
			return -1
	else:
		return -1

def perp_indv(article):
	sentence_tokens=sent_tokenize(article)

	set_of_perp_sentences=set()

	for single_sentence in sentence_tokens:
	
		for phrase in list_of_keywords:
			ratio=fuzz.token_set_ratio(single_sentence,phrase)
			if ratio>90:
				#print("RATIO:",ratio,single_sentence,"PHRASE:",phrase)		
				x=nlp.pos_tag(single_sentence.lower())
				set_of_perp_sentences.add(single_sentence)


	list_of_perp_indv=[]
	list_of_sentences=[]
	for single_sentence in set_of_perp_sentences:

			
				x=nlp.pos_tag(single_sentence.lower())
				#print (single_sentence)
				#print x

				gram= r'''
				G1: {<RB>*<NN|NN.>*<IN>*<JJ>*<NN|NN.>+<VB.>+<DT>*<VB.>*<CD>*<IN>*<DT>*<JJ>*<NN|NN.>+}
				G2: {<CD>*<IN>*<DT>*<NN|NN.>+<WP><VB.>+<IN><VB.>*<IN>*<DT>*<NN|NN.>+}
				G3: {<NN|NN.>*<VB.>+<IN>+<CD>*<JJ>*<NN|NN.>+}
				G4: {<CD>*<NN|NN.>+<VB.>+<JJ>*<DT>*<NN|NN.>*<IN>+<NN|NN.>+}
				G5: {<CD>*<IN><DT>*<NN|NN.>+<VB.>+<NN|NN.>+}
				G6: {<JJ>*<NN|NN.>+<VB.>+<RP>*<DT>*<NN|NN.>+<IN>*}
				
				G8: {<NN|NN.>+<VB.>+<DT>*<NN|NN.>+<IN>+<DT>*<NN|NN.>+}
				G9: {<CD>*<RB>*<NN|NN.>+<VB.>+<JJ>*<IN>+}
							'''

				chunked_text = nltk.RegexpParser(gram)
				tree = chunked_text.parse(x)

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G1'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))
			
	
				for subtree in tree.subtrees(filter = lambda t: t.label()=='G2'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G3'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G4'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G5'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G6'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G7'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G8'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G9'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))

				for subtree in tree.subtrees(filter = lambda t: t.label()=='G10'):
					list_of_sentences.append(" ".join([a for (a,b) in subtree.leaves()]))
	

	nounphrases1=[]
	nounphrases=[]
	nounphrases2=[]
	for s in set(list_of_sentences):
		doc = ennlp(unicode(s))

		for np in doc.noun_chunks:
			nounphrases1.append(np)
			nounphrases2 = list(set(nounphrases1))
		for i in nounphrases2:
			if i:
				nounphrases.append(i)
	
	#trying to remove the noun phrases which contains verbs
	updated_nounphrases=[]
	for np in set(nounphrases):
		x=nlp.pos_tag(str(np))
		#print("pos",x)
		flag=1		
		for i in range(0,len(x)):
			if x[i][0].encode('utf-8') in indv_list:
				flag=0
				break
		if flag==1:
			updated_nounphrases.append(np)
		
		
	
	finalans=[]
	for i in set(updated_nounphrases):	
		if(str(i)!=" "):
				if(len(finalans)!=0):
					if "the" in str(i):
						i=str(i).replace("the ","")
						finalans.append(str(i))
					else:
						finalans.append(str(i))
				else:
					if "the" in str(i):
						i=str(i).replace("the ","")
						finalans.append(str(i))
					else:
						finalans.append(str(i))
	if (len(finalans)!=0):
		#print finalans
		return list(set(finalans))
	else:
		#print "-"
		return -1


def features(article, article_id):

	fout.write("ID:             {}\n".format(article_id))

	incident = incident_value(article_id)
	fout.write("INCIDENT:       {}\n".format(incident.upper()))

	weapon = weapons(article)
	if weapon == -1:
		fout.write("WEAPON:         {}\n".format("-"))
	else:
		for i in range(len(weapon)):
			if i == 0:
				fout.write("WEAPON:         {}\n".format(weapon[i].upper()))
			else:
				fout.write("                {}\n".format(weapon[i].upper()))

	perp = perp_indv(article)
	if perp == -1:
		fout.write("PERP INDIV:     {}\n".format("-"))
	else:
		for i in range(len(perp)):
			if i == 0:
				fout.write("PERP INDIV:     {}\n".format(perp[i].upper()))
			else:
				fout.write("                {}\n".format(perp[i].upper()))


	org = organisation(article)
	fout.write("PERP ORG:       {}\n".format(org.upper()))

	tar = target(article)
	if tar == -1:
		fout.write("TARGET:         {}\n".format("-"))
	else:
		for i in range(len(tar)):
			if i == 0:
				fout.write("TARGET:         {}\n".format(tar[i].upper()))
			else:
				fout.write("                {}\n".format(tar[i].upper()))

	vict = victim(article)
	if vict == -1:
		fout.write("VICTIM:         {}\n".format("-"))
	else:
		for i in range(len(vict)):
			if i == 0:
				fout.write("VICTIM:         {}\n".format(vict[i].upper()))
			else:
				fout.write("                {}\n".format(vict[i].upper()))

	fout.write("\n")
'''
	tar = target(article)
	if tar == -1:
		print("TARGET:         {}\n".format("-"))
	else:
		for i in range(len(tar)):
			if i == 0:
				print("TARGET:         {}\n".format(tar[i].upper()))
			else:
				print("                {}\n".format(tar[i].upper()))
'''


def extract_article(unique_id, corpus):
	for i in range(len(corpus)):
		match = search(r"DEV-MUC3-[0-9]{4}",corpus[i]) or search(r"TST1-MUC3-[0-9]{4}",corpus[i]) or search(r"TST2-MUC4-[0-9]{4}",corpus[i])

		article = " "

		if match:
			for j in range(i+1, len(corpus)):
				idMatch = search(r"DEV-MUC3-[0-9]{4}",corpus[j]) or search("TST1-MUC3-[0-9]{4}",corpus[j]) or search("TST2-MUC4-[0-9]{4}",corpus[j])
				
				if idMatch:
					if idMatch.group() in unique_id:
						break
				else:
					article = article + "".join(corpus[j].lower()) + " "

			features(article, match.group())



unique_id, corpus = extract_id(sys.argv[1])
weapon_list = create_weapons()
target_list = create_target()
victimlist = create_victim()
indv_list = create_skip_indv()
skip_list = create_skip()
general_indiv = general()
incident_dict=incidents()
final_org_name = organisation_names("organisation.txt")				
extract_article(unique_id, corpus)
fout.close()
