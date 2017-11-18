
#勉強する教材

オセロAIに関して私が理解できる本は唯一これだけでした

<a href = "http://shop.ohmsha.co.jp/shopdetail/000000004775/">
<img src="https://qiita-image-store.s3.amazonaws.com/0/142847/8454f7c0-443d-16bd-dad7-5e6d3b1a38bb.jpeg" width=40%> </a>

この本の **第６章 強化学習ー三目並べに強いコンピューターを育てる**
を元にオセロAIを作成します。

この他に参考になる書籍をご存知の方ご教示ください


#まずは教科書通り...と言いたいところですが

この本の **第６章 強化学習ー三目並べに強いコンピューターを育てる**
では下記の点が気に食わないので、改良したものを 再始動の初期状態とします。

1. python2 を python3 に変換する

2. **RL-Glue** という仕組みは、削除する (私には難しい)


#学習するには

```text:
$ cd train-chainerai
$ python experiment.py
```

#下図の勝率図で、正しく学習出来ていることを確認
![](https://github.com/sasaco/train-chainerai/blob/%231/percentages.png)

