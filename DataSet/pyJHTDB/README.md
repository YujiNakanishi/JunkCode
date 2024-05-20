### 環境構築(2024/4/18)
dockerの[JHTDB]コンテナ上で構築した。公式サイトに書かれている、pipによるインストールは失敗した。そこで、下記の通りgitを介す。  
```
git clone https://github.com/idies/pyJHTDB.git
cd pyJHTDB
python update_turblib.py
pip install --upgrade ./
```
当然ながらgit環境は必要。また、「クローンされたフォルダを信用してよいか？」みたいなことをupdate_turblib.pyのときにgitに聞かれる。信用してよい場合はこれ、みたいなコマンドを提示されるので、それをコピペして実行。  
上記コマンド実行後はpyJHTDBリポジトリを消しても構わない。