import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def main():
    
	with open('review.csv') as csv_file:
            reader = csv.reader(csv_file,delimiter=",",quotechar='"')
            reader.next()
            data =[]
            target = []
            for row in reader:
                # skip missing data
                if row[0] and row[1]:
                    data.append(row[0])
                    #print row[0]
                    target.append(row[1])
                    #print row[1]
                    #raw_input('>')
	    
	    
	#print data
	
	count_vectorizer = CountVectorizer(ngram_range=(1,1))
	data = count_vectorizer.fit_transform(data)
	#print data
	
	data1 = {"excellent This guy knows how to entertain an audience!","It is boring and interesting interesting","From the first scene you are given clues as to what may be going on here. It becomes more and more obvious as the story rolls on. The acting is excellent throughout and these actors touch your soul. Even though I knew what was going to happen I was extremely puzzled by the motive. I'm still puzzled as to why Ben did what he did. We could see in his face second thoughts, but the ultimate sacrifice seemed to go against his emotion and feelings. It was a very interesting and touching story but it left me confused. Maybe that was the point of the film. I did like the film and Wil Smith can wrack up another good film choice. This guy knows how to entertain an audience!","It would be impossible to sum up all the stuff that sucks about this film, so I'll break it down into what I remember most strongly: a man in an ingeniously fake-looking polar bear costume (funnier than the bear from Hercules in New York); an extra with the most unnatural laugh you're ever likely to hear; an ex-dope addict martian with tics; kid actors who make sure every syllable of their lines are slowly and caaarreee-fulll-yyy prrooo-noun-ceeed; a newspaper headline stating that Santa's been kidnaped, and a giant robot. Yes, you read that right. A giant robot.The worst acting job in here must be when Mother Claus and her elves have been frozen by the Martians weapons. Could they be more trembling? I know this was the sixties and everyone was doped up, but still.","Spirited Away' is the first Miyazaki I have seen, but from this stupendous film I can tell he is a master storyteller. A hallmark of a good storyteller is making the audience empathise or pull them into the shoes of the central character. Miyazaki does this brilliantly in 'Spirited Away'. During the first fifteen minutes we have no idea what is going on. Neither does the main character Chihiro. We discover the world as Chihiro does and it's truly amazing to watch. But Miyazaki doesn't seem to treat this world as something amazing. The world is filmed just like our workaday world would. The inhabitants of the world go about their daily business as usual as full with apathy as us normal folks. Places and buildings are not greeted by towering establishing shots and majestic music. The fact that this place is amazing doesn't seem to concern Miyazaki.What do however, are the characters. Miyazaki lingers upon the characters as if they were actors. He infixes his animated actors with such subtleties that I have never seen, even from animation giants Pixar. Twenty minutes into this film and I completely forgot these were animated characters; I started to care for them like they were living and breathing. Miyazaki treats the modest achievements of Chihiro with unashamed bombast. The uplifting scene where she cleanses the River God is accompanied by stirring music and is as exciting as watching gladiatorial combatants fight. Of course, by giving the audience developed characters to care about, the action and conflicts will always be more exciting, terrifying and uplifting than normal, generic action scenes. ","I simply love this movie. I also love the Ramones, so I am sorta biased to begin with in the first place. There isn't a lot of critical praise to give this film, either you like it or you don't. I think it's a great cult movie."}
	data1 = count_vectorizer.transform(data1)
	#print data1
	data2 ={"I simply love this movie. I also love the Ramones, so I am sorta biased to begin with in the first place. There isn't a lot of critical praise to give this film, either you like it or you don't. I think it's a great cult movie.","I simply hate this movie. I also hate the Ramones, so I am sorta biased to begin with in the first place. There isn't a lot of critical condemn to give this film, either you like it or you don't. I think it's a unenthusiastic cult movie."}
	data2 = count_vectorizer.transform(data2)
	data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=43)
	
	classifier = BernoulliNB().fit(data_train,target_train)
	
	predicted = classifier.predict(data_test)
	predicted1 = classifier.predict(data1)
	predictedProbability = classifier.predict_proba(data1)
	print predicted1
	print predictedProbability
	print "------------------------"
	
	predicted2 = classifier.predict(data2)
	predictedProbability2 = classifier.predict_proba(data2)
	print predicted2[0]
	print predictedProbability2[0]
	#print data2[0]
	print predicted2[1]
	print predictedProbability2[1]
	#print data2[1]
	
	print "------------------------"
	data2 ={"I really enjoyed this documentary about Kenny and Spencer's attempt to pitch The Dawn. Was a great look at how outsiders try to get to the inside to make it big. <br /><br />The story was put together well and organized in an interesting manner that made the film flow well. Certainly worth a watch. My only complaint is that their appeared to be no closure. Perhaps that is part of the point. We expect it but in reality that is not what happened (or usually happens).<br /><br />The film is also a great way to see the personality of Kenny and Spencer outside of their Canadian television show. You can see a bit of what is yet to come. <br /><br />I look forward to a chance to see The Papal Chase.","I really hated this documentary about Kenny and Spencer's attempt to pitch The Dawn. Was a unenthusiastic look at how outsiders try to get to the inside to make it big. <br /><br />The story was put together badly and unorganized in an uninteresting manner that made the film flow badly. Certainly not worth a watch. My only complaint is that their appeared to be closure. Perhaps that is part of the point. We expect it but in reality that is not what happened (or usually happens).<br /><br />The film is also a unenthusiastic way to see the personality of Kenny and Spencer outside of their Canadian television show. You can see a bit of what is yet to come. <br /><br />I don't look forward to a chance to see The Papal Chase."}
	data2 = count_vectorizer.transform(data2)
	predicted2 = classifier.predict(data2)
	predictedProbability2 = classifier.predict_proba(data2)
	print predicted2[0]
	print predictedProbability2[0]
	#print data2[0]
	print predicted2[1]
	print predictedProbability2[1]
	
	print "------------------------"
	data2 ={"From the first scene you are given clues as to what may be going on here. It becomes more and more obvious as the story rolls on. The acting is excellent throughout and these actors touch your soul. Even though I knew what was going to happen I was extremely puzzled by the motive. I'm still puzzled as to why Ben did what he did. We could see in his face second thoughts, but the ultimate sacrifice seemed to go against his emotion and feelings. It was a very interesting and touching story but it left me confused. Maybe that was the point of the film. I did like the film and Wil Smith can wrack up another good film choice. This guy knows how to entertain an audience!","From the first scene you are left guessing as to what may be going on here. It becomes more and more unobvious as the story rolls on. The acting is poor throughout and these actors don't touch your soul. Even though I knew what was going to happen I was extremely puzzled by the motive. I'm still puzzled as to why Ben did what he did. We could see in his face second thoughts, but the ultimate sacrifice seemed to go against his emotion and feelings. It was a very uninteresting and untouching story but it left me confused. Maybe that was the point of the film. I did dislike the film and Wil Smith can wrack up another bad film choice. This guy knows how not to entertain an audience!"}
	data2 = count_vectorizer.transform(data2)
	predicted2 = classifier.predict(data2)
	predictedProbability2 = classifier.predict_proba(data2)
	print predicted2[0]
	print predictedProbability2[0]
	#print data2[0]
	print predicted2[1]
	print predictedProbability2[1]
	
	
	print "------------------------"
	data2 ={"This movie lacked... everything: story, acting, surprise, ingenuity and a soul. Fifteen minutes in, I was staring at the screen saying, How could all of these guys get together and consider themselves friends (even without the girl)? Another fifteen minutes in, I was praying for as much Amanda Peet as possible. When a bad movie quietly rears it's ugly head, eye candy is a nice consolation. But there wasn't much of that! Cheated on all fronts! ","This movie has ... everything: story, acting, surprise, ingenuity and a soul. Fifteen minutes in, I was staring at the screen saying, How could all of these guys get together and consider themselves friends (even without the girl)? Another fifteen minutes in, I was praying for as much Amanda Peet as possible. When a good movie quietly rears it's beautiful head, eye candy is a nice consolation. But there much of that! good on all fronts! "}
	data2 = count_vectorizer.transform(data2)
	predicted2 = classifier.predict(data2)
	predictedProbability2 = classifier.predict_proba(data2)
	print predicted2[0]
	print predictedProbability2[0]
	#print data2[0]
	print predicted2[1]
	print predictedProbability2[1]
	
	
	





main()
