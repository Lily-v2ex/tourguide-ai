# constants.py
"""This module defines project-level constants."""

ATTRACTION_SYSTEM_MESSAGE = (
    "Given the input attraction, "
    "generate a search query to write an INTRODUCTION about it."
    "The INTRODUCTION should be playful and interactive."
    "The INTRODUCTION must follow the below rules."
    "The first paragraph is background introduction, "
    "for example, the meaning of the name, "
    "the attraction location, "
    "the importance."
    "The second paragraph talks about its size, "
    "architecture style, "
    "and architecture characteristics."
    "Then list the main buildings "
    "in the form of bulleted list with any preamble text."
    "The third Paragraph lists the titles "
    "that the attraction has earned."
    "EXAMPLE INPUT"
    "Qingjing Mosque"
    "EXAMPLE INTRODUCTION"
    "Qingjing Mosque, originally known as Shengyou Temple, "
    "also know as Aisuhab Grand Mosque,"
    "is located in Tumen Street, Quanzhou City, Fujian Province."
    "It is the oldest existing Islamic temple "
    "founded by Arab Muslims in China."
    "It was built in the Northern Song Dynasty "
    "in the second year of Dazhong Xiangfu(AD 1009)."
    "Qingjing Mosque covers an area of 2,184 square meters, "
    "and the whole is a stone building,"
    "imitating the architectural form of "
    "the Islamic Chapel in Damascus, Syria"
    "It has the characteristics of large dispersion "
    "and small concentration in the functional"
    "space of Islamic mosques."
    "The main buildings than remain are the gatehouse,"
    " the worship hall, the Mingshan Hall."
    "In 1961, Qingjing Mosque was listed "
    "by the State Council as the first batch of national"
    "key cultural relics protection units."
    "In the 90s of the 20th century, "
    "it was listed as the only Islamic mosque selected "
    "among the Top Ten Famous Temples in China. "
    "Together with Yangzhou Xianhe Temple, Guangzhou Huaisheng "
    "Temple and Hanzhou Phoenix Temple."
    "It is known as the four ancient temples of Islam in China, "
    "and its establishment is one of "
    "important historical sites of overseas exchanges in Quanzhou."
)

OTHER_SYSTEM_MESSAGE = "Given the input, generate a search query to search result."
