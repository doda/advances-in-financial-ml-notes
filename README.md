# Notes & Solutions to Advances in Financial Machine Learning

I've been playing around with various trading strategies recently and one day stumbled upon [Advances in Financial Machine Learning my Marcos Lopez de Prado](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089):

![](https://i.imgur.com/QZYNwxx.jpg)


Early on in the book, Marcos writes:

>Investment management is one of the most multi-disciplinary areas of research, and this book reflects that fact. Understanding the various sections requires a practical knowledge of ML, market microstructure, portfolio management, mathematical finance, statistics, econometrics, linear algebra, convex optimization, discrete math, signal processing, information theory, object-oriented programming, parallel processing and supercomputing.

And so while I can't quite claim all these competences (I would consider myself a beginner in 10 out of these 14 areas), I was still super-curious how one might go about algorithmic trading if they did and so embarked upon solving the exercises presented at the end of each chapter. The notebooks above contain these attempts at solutions.

Thanks:
--------------------------------

- **Huge thanks to Marcos for writing this book**. It's hard to explain how valuable it is, especially for someone like me. Marcos has synthesized 20 years of experience in financial mathematics and computer science into the most important & effective areas and then provided great code and guidance within them. In my view it stands completely alone in an industry shrouded in secrecy and elitism. He has helped me upgrade my thinking and toolkit 10x within just 2 weeks of working through the material. There were too many a-ha moments to count and I now have a much better picture of what I need to further learn & do to succeed in algorithmic trading. ❤️ 
- Many thanks also to https://github.com/hudson-and-thames/ for their [solutions](https://github.com/hudson-and-thames/research) and [mlfinlab package](https://github.com/hudson-and-thames/mlfinlab). Sometimes when nothing seemed to be working I was able to fall back on their solutions (where available) and implementations to sanity-check whether the bug was in my data, my understanding of the problem, my code or Marcos' code. (it was mostly #2 or #3)

Notes:
--------------------------------

- All of the questions and most of the code was transcribed from the book, slightly modified in places and made PEP-8 compliant. Most everything `camelCase` is Marcos' code, while `snake_case` is mine.
- While I have a lot of experience in Python, I do not in finance or math, so there are likely bugs in my results somewhere :) This was exacerbated by the ridiculous pace I put on myself to work through the chapters (about 1 per day).

**Current state of completion:**

Begun or Done
- [ ] `[3/5]` Chapter 2 - Financial Data Structures
- [x] `[5/5]` Chapter 3 - Meta-Labeling
- [x] `[7/7]` Chapter 4 - Sample Weights
- [ ] `[5/6]` Chapter 5 - Fractionally Differentiated Features
- [x] `[5/5]` Chapter 6 - Ensemble Methods
- [x] `[5/5]` Chapter 7 - Cross-Validation in Finance
- [x] `[5/5]` Chapter 8 - Feature Importance
- [x] `[6/6]` Chapter 9 - Hyper-Parameter Tuning with Cross-Validation
- [ ] `[4/7]` Chapter 10 - Bet Sizing
- [x] `[5/5]` Chapter 11 - The Dangers of Backtesting
- [x] `[5/5]` Chapter 12 - Backtesting through Cross-Validation
- [ ] `[2/6]` Chapter 13 - Backtesting on Synthetic Data
- [x] `[7/7]` Chapter 14 - Backtest Statistics
- [ ] `[4/6]` Chapter 15 - Understanding Strategy Risk
- [ ] `[3/5]` Chapter 16 - Machine Learning Asset Allocation

Open 
- [ ] `[0/5]` Chapter 17 - Structural Breaks
- [ ] `[0/5]` Chapter 18 - Entropy Features
- [ ] `[0/12]` Chapter 19 - Microstructural Effects
- [ ] `[0/6]` Chapter 20 - Multiprocessing and Vectorization


