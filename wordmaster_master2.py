
"""
Created on Thu Feb 10 09:25:33 2022

@author: jhoffman
"""

import tkinter as tk
import os

global guess, rect, text, rect_c, letter, let_pos, not_pos

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'wordle.txt')
wordtext = open(filename,'r')

temp_word = wordtext.readline()
word_list = []
while temp_word != '':
    word_list.append(temp_word[-6:-1])
    temp_word = wordtext.readline()


def create_list():
    temp_list = []
    for i in word_list:
        temp_list.append(i)
    return temp_list

def letter_frequency(current_words):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    alph_freq = [0]*26
    for i in current_words:
        for j in alphabet:
            if j in i and j not in let_pos:
                alph_freq[alphabet.index(j)] += 1
    return alph_freq

def letter_frequency_place(current_words):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    alph_freq = [0]*26
    for j in range(26):
        alph_freq[j] = [0]*5
    for i in current_words:
        for n in range(len(i)):
            if let_pos[n] == '':
                l = i[n]
                alph_freq[alphabet.index(l)][n] += 1

    return alph_freq

def word_value(word, alph_freq, freq_place):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    unique = ''
    for i in alphabet:
        if i in word:
            unique += i
    sum = 0
    for i in unique:
        sum += alph_freq[alphabet.index(i)]
    sum2 = 0
    for i in unique:
        if i not in let_pos:     
            sum2 += freq_place[alphabet.index(i)][word.index(i)]
    return sum + 3*sum2

def guess_word(all_words):
    words = cut_words(all_words)
    search = search_words(all_words)
    letter_freq = letter_frequency(words)
    letter_freq_p = letter_frequency_place(words)
    high_word = ''
    high_value = 0
    for w in words:
        temp = word_value(w, letter_freq, letter_freq_p)
        if temp >= high_value:
            high_value = temp
            high_word = w
    if len(words) > 2:  
        for w in search:
            temp = word_value(w, letter_freq, letter_freq_p)
            if temp > high_value:
                high_value = temp
                high_word = w   
    print(words)
    print(let_pos)
    print(not_pos)

    if high_word == '':
         return words[0]

    return high_word

root= tk.Tk()

guess = 0

canvas1 = tk.Canvas(root, width = 300, height = 420,  relief = 'raised')
canvas1.pack()

canvas1.create_text(150, 35, text="WORDMASTER MASTER", fill="black", font=('Helvetica 15 bold'))
rect = ['']*30
text = ['']*30
rect_c = ['g']*30
letter = ['']*30
let_pos = ['']*5
not_pos = ['']*5
x1,x2,y1,y2 = 46,86,70,110
for i in range(30):
    rect[i] = canvas1.create_rectangle(x1,y1,x2,y2, fill = 'gray')
    text[i] = canvas1.create_text(x1 +20, y1+20, text="", fill="black", font=('Helvetica 10'))
    x1 += 43
    x2 += 43
    if (i+1)%5 == 0:
        x1, x2 = 46,86
        y1 += 43
        y2 += 43
    
canvas1.create_rectangle(60,360,145,400, fill = 'gray')
canvas1.create_text(102, 380, text="New Game", fill="black", font=('Helvetica 10'))

canvas1.create_rectangle(155,360,240,400, fill = 'gray')
canvas1.create_text(198, 380, text="Next Word", fill="black", font=('Helvetica 10'))  

def push_word(word):
    global guess, letter, text
    if len(word) < 5: 
        return
    yellow = ''
    for i in range(5):
        yellow += not_pos[i]
    for i in range(5):
        canvas1.itemconfig(text[i+guess*5], text = word[i])
        letter[i + guess*5] = word[i]
        if word[i] in yellow: 
            rect_c[i + guess*5] = 'y'
            canvas1.itemconfig(rect[i+guess*5], fill = 'orange')
        if word[i] == let_pos[i]:
            rect_c[i + guess*5] = 'e'
            canvas1.itemconfig(rect[i+guess*5], fill = 'green')
        
    
def cut_words(words):
    temp = []
    yellow = ""
    gray = ""
    for i in range(5):
        yellow += not_pos[i]
    for i in range(30):
        if rect_c[i] == 'g' and letter[i] not in yellow and letter[i] not in let_pos:
            gray += letter[i]
    for w in words:
        t = True
        for l in yellow:
            if l not in w and l not in let_pos: t = False
        for i in range(5):
            if w[i] in gray: t = False
            if w[i] in not_pos[i]: t = False
            if let_pos[i] != '' and w[i] != let_pos[i]: t = False
        if t: temp.append(w)  
    return temp

def search_words(words):
    temp = []
    yellow = ""
    gray = ""
    temp_y = ''
    for i in range(30):
        if rect_c[i] == 'g' or rect_c[i] == 'e':
            gray += letter[i]
    for i in range(5):
         temp_y += not_pos[i]
    for i in temp_y:
        if i not in let_pos:
            yellow+=i
    for w in words:
        t = True
        for l in yellow:
            if l not in w: t = False
        for i in range(5):
            if w[i] in not_pos[i]: t = False
            if w[i] == let_pos[i]: t = False
            if let_pos[i] != '' and w[i] in yellow: t = False
        if t: temp.append(w)
    return temp

def new():
    global guess, rect_c, letter, let_pos, not_pos
    guess = 0
    rect_c = ['g']*30
    letter = ['']*30
    let_pos = ['']*5
    not_pos = ['']*5
    
    for i in range(30):
        canvas1.itemconfig(rect[i], fill = 'grey')
        canvas1.itemconfig(text[i], text = '')
        
    word = guess_word(create_list())
    push_word(word)

def next():
    global guess
    for i in range(5):
        if rect_c[i + guess*5] != 'g':
            if rect_c[i + guess*5] == 'y':
                not_pos[i] += letter[i + guess*5]
            if rect_c[i + guess*5] == 'e':
                let_pos[i] = letter[i + guess*5]

    guess += 1
    push_word(guess_word(create_list()))


def click(event):
    box = -1
 
    x1,x2,y1,y2 = 46,86,70,110
    for i in range(30):
        if guess == int(i/5) and event.x > x1 and event.x < x2 and event.y > y1 and event.y < y2: box = i
        x1 += 43
        x2 += 43
        if (i+1)%5 == 0:
            x1, x2 = 46,86
            y1 += 43
            y2 += 43 
    
    if event.x > 60 and event.x < 145 and event.y > 360 and event.y < 400:
        new()

    if event.x > 155 and event.x < 240 and event.y > 360 and event.y < 400:
        next()
    

    if box >= 0 and box < 30:
        if rect_c[box] == 'g':
            canvas1.itemconfig( rect[box], fill = 'orange')
            rect_c[box] = 'y'
        elif rect_c[box] == 'y':
            canvas1.itemconfig( rect[box], fill = 'green')
            rect_c[box] = 'e'
        else:
            canvas1.itemconfig( rect[box], fill = 'gray')
            rect_c[box] = 'g'

new()
canvas1.bind('<Button>', click)
root.mainloop()