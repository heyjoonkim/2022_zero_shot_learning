

a = {
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "logprobs": {
        "text_offset": [
          0,
          1,
          4,
          8,
          10,
          11,
          15,
          21,
          24,
          28,
          34,
          46,
          50,
          52,
          53,
          57,
          62,
          65,
          70,
          77,
          80,
          84,
          87,
          88,
          98,
          101,
          105,
          117,
          121,
          123,
          134,
          135,
          144,
          146,
          158,
          162,
          164,
          168,
          171,
          180,
          181,
          185,
          188,
          194,
          195,
          202,
          203
        ],
        "token_logprobs": [
          "null",
          -7.042436,
          -6.795585,
          -0.024955748,
          -1.080815,
          -1.5058265,
          -1.5284399,
          -0.011376138,
          -1.2795904,
          -1.3007622,
          -1.593323,
          -0.05339957,
          -4.3347598e-05,
          -0.20602082,
          -2.2592866,
          -1.0897534,
          -1.4622334,
          -0.13626844,
          -0.026322806,
          -0.966456,
          -1.6600255,
          -0.031023128,
          -1.4626012,
          -1.7315315,
          -0.0019013024,
          -1.543206,
          -2.4843621,
          -2.3154106,
          -1.2351458e-05,
          -0.64988214,
          -0.2526543,
          -14.247086,
          -5.6510167,
          -4.987421,
          -0.24853286,
          -0.00015982577,
          -3.7521439,
          -3.0603426,
          -1.1599761,
          -3.0722716,
          -8.726408,
          -1.2613617,
          -0.48334393,
          -0.94630545,
          -6.903828,
          -1.4424948,
          -1.5178705
        ],
        "tokens": [
          "D",
          "ana",
          " Ree",
          "ve",
          ",",
          " the",
          " widow",
          " of",
          " the",
          " actor",
          " Christopher",
          " Ree",
          "ve",
          ",",
          " has",
          " died",
          " of",
          " lung",
          " cancer",
          " at",
          " age",
          " 44",
          ",",
          " according",
          " to",
          " the",
          " Christopher",
          " Ree",
          "ve",
          " Foundation",
          ".",
          " Question",
          " :",
          " Christopher",
          " Ree",
          "ve",
          " had",
          " an",
          " accident",
          ".",
          "True",
          " or",
          " False",
          "?",
          " answer",
          ":",
          " True"
        ],
        "top_logprobs": [
          "null",
          {
            "-": -3.1623816
          },
          {
            " White": -2.9146135
          },
          {
            "ve": -0.024955748
          },
          {
            ",": -1.080815
          },
          {
            " the": -1.5058265
          },
          {
            " wife": -0.99531686
          },
          {
            " of": -0.011376138
          },
          {
            " the": -1.2795904
          },
          {
            " late": -0.8018761
          },
          {
            " who": -0.7131388
          },
          {
            " Ree": -0.05339957
          },
          {
            "ve": -4.3347598e-05
          },
          {
            ",": -0.20602082
          },
          {
            " who": -1.7520119
          },
          {
            " died": -1.0897534
          },
          {
            " of": -1.4622334
          },
          {
            " lung": -0.13626844
          },
          {
            " cancer": -0.026322806
          },
          {
            " at": -0.966456
          },
          {
            " the": -0.98803616
          },
          {
            " 44": -0.031023128
          },
          {
            ".": -0.30661428
          },
          {
            " the": -1.6533568
          },
          {
            " to": -0.0019013024
          },
          {
            " a": -1.5314147
          },
          {
            " Associated": -0.5405509
          },
          {
            " &": -0.68452847
          },
          {
            "ve": -1.2351458e-05
          },
          {
            " Foundation": -0.64988214
          },
          {
            ".": -0.2526543
          },
          {
            "\n": -0.5858581
          },
          {
            ":": -0.73450536
          },
          {
            " What": -2.1086183
          },
          {
            " Ree": -0.24853286
          },
          {
            "ve": -0.00015982577
          },
          {
            "'s": -1.7840196
          },
          {
            " a": -1.1729907
          },
          {
            " accident": -1.1599761
          },
          {
            " in": -1.7360824
          },
          {
            " He": -1.849806
          },
          {
            " or": -1.2613617
          },
          {
            " False": -0.48334393
          },
          {
            "?": -0.94630545
          },
          {
            " Answer": -1.2072545
          },
          {
            ":": -1.4424948
          },
          {
            " True": -1.5178705
          }
        ]
      },
      "text": "Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44, according to the Christopher Reeve Foundation. Question : Christopher Reeve had an accident.True or False? answer: True"
    }
  ],
  "created": 1648106878,
  "id": "cmpl-4pCAwG5VEwYY8iRdgcHmM7wJ9BIfO",
  "model": "davinci:2020-05-03",
  "object": "text_completion"
}

data = a['choices'][0]
logprobs = data['logprobs']
token_logprobs = logprobs['token_logprobs']
tokens = logprobs['tokens']

l = len(token_logprobs)

print('token_logprobs', len(token_logprobs))
print('tokens', len(tokens))
print('-' * 30)

for i in range(l):
    print(token_logprobs[i], '->', tokens[i])