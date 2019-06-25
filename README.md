# SeqWORDS package
SeqWORDS is an unsupervised Chinese word segmentation method, which not demand a dictionary in hand. This package is an implementation of SeqWORDS algorithm on python.
## Installation
To install this package, execute command below in terminal.
```bash
pip install SeqWORDS
```
## Usage
```python
import SeqWORDS
SeqWORDS.cut(corpus)
```
## Parameter
| parameter      | type | description                                      |
| -------------- | ---  | ------------------------------------------------ |
| `tuaL`         | Int  | the output of dox as a parsed JSON object        |
| `tuaF`         | Int  | whether to output a readme or just docs          |
| `useProb1`     | Int  | a parsed package.json                            |
| `useProb2`     | Int  | whether to output a travis badge along with docs |
| `connectThld`  | Int  | whether to output a travis badge along with docs |
| `travis`       | Int  | whether to output a travis badge along with docs |
| `travis`       | Int  | whether to output a travis badge along with docs |
| `travis`       | Int  | whether to output a travis badge along with docs |
## Example
### Story of Stone
Story of Stone, also called Dream of the Red Chamber, composed by Xueqin Cao in 18th century during the Qing dynasty. The novel features in massive number of characters.
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
