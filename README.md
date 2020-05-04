# ProRob_Code
 詳解確率ロボティクス勉強用

## 環境
- Windows 10
- Python 3.7
- Anaconda
- Atom
  - Hydorogen

### アニメーションについて
AtomのHydorogenでコード記述＋実行.
matplotlib の nbagg の使用が上手くいかなかった．
代わりに以下のコードを先頭に記入している．
別ウインド表示で，アニメーションが動作する．

~~~
%matplotlib qt
~~~

### 自作モジュールのimportについて
``%matplotlib qt``は本来Jupyter Notebookでの表記であるためか，.pyファイルをそのままimportできなかった．したがって，以下に書き換えた．

~~~
get_ipython().run_line_magic('matplotlib', 'qt')
~~~

また，Atomで自作モジュールを含んだフォルダがプロジェクト内に含まれていると，import時にエラーが表示された.
自作モジュールを含むフォルダがプロジェクトフォルダに含まれないように調整する必要があった．
