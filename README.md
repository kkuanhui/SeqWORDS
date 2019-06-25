# SeqWORDS: An Unsupervised Chinese Word Segmentation package
This is Python SeqWORDS package. SeqWORDS is an unsupervised Chinese word segmentation method, which not demand a dictionary in hand.
## Installation
```bash
pip install SeqWORDS
```
## Usage
```python
import SeqWORDS

file = open(example.txt, mode = r)
corpus = file.read()
file.close()

SeqWORDS.cut(corpus)
```
## Example
### Introduction of Story of Stone
Story of Stone, also called Dream of the Red Chamber, composed by Xueqin Cao in 18th century during the Qing dynasty.
### Results
Below is word cloud, it shows the most frequent words. "寶玉" is the biggest one amoung of all cloud. 
<figure>
<img src="SeqWORDS_cloud.png"
    alt="SeqWORDS_cloud"
    style="float: left; margin-right: 10px;" />
<figcaption> word cloud SeqWORDS</figcaption>
</figure>

Below is PCA of word vectors. The plot containing 51 words includes "寶玉" and 50 words that most relative to "寶玉". Amoung these words, there 42 names has great relativity to "寶玉". 

<figure>
<img src="010_word2vec_SeqWORDS.png"
    alt="010_word2vec_SeqWORDS"
    style="float: left; margin-right: 10px;" />
<figcaption> SeqWORDS</figcaption>
</figure>

## License
[MIT](https://choosealicense.com/licenses/mit/)
