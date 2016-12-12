from __future__ import division
import numpy as np
# "cimport" は、 numpy モジュールを使うコードのコンパイル時に必要
# な情報を import するのに使います。
# (この情報は numpy.pxd に入っています。現状、 numpy.pxd は Cython
# ディストリビューションに入っています)
cimport numpy as np
# 変数 DTYPE を使って、アレイのデータ型に手を加えます。DTYPE には、
# NumPy のランタイム型情報オブジェクト (runtime type info object)
# を代入します。
DTYPE = np.int
# "ctypedef" は、コンパイル時に決定した DTYPE_t で参照できるよ
# うにします。numpy モジュールのデータ型には、すべて _t という
# サフィックスのついたコンパイル時用の型があります。
ctypedef np.int_t DTYPE_t
# "def" は引数の型指定を行えますが、戻り値の型を指定できません。
# "def" 関数型の引数の型は、関数に入るときに動的にチェックされます。
#
# アレイ f, g, h は "np.ndarray" インスタンス型に型付けされていま
# す。その効果は a) 関数の引数が本当に NumPy アレイかどうかチェッ
# クが入ることと、 b) f.shape[0] のようなアトリビュートアクセスの
# 一部が格段に効率的になること、だけです (この例の中では、どちらも
# さして意味はありません)。
from cython.parallel cimport prange
from libc.math cimport exp, ceil
from libc.stdlib cimport rand, RAND_MAX
# 並列計算関連のモジュールをインポート
cimport cython
@cython.boundscheck(False) # 関数全体で境界チェックを無効化
def parallel_convolve(np.ndarray[DTYPE_t, ndim=2] f, np.ndarray[DTYPE_t, ndim=2] g):
    # f は画像で、 (v, w) でインデクスする
    # g はフィルタカーネルで、 (s, t) でインデクスする
    #   ディメンジョンは奇数でなくてはならない
    # h は出力画像で、 (x, y) でインデクスする
    #   クロップしない
    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")
    # smid と tmid は中心ピクセルからエッジまでのピクセル数、
    # つまり 5x5 のフィルタならどちらも 2
    #
    # 出力画像のサイズは入力画像の両端に smid と tmid を足した
    # 値となる
    assert f.dtype == DTYPE and g.dtype == DTYPE
    # "cdef" キーワードは、関数の中で変数の型を宣言するのにも使い
    # ます。 "cdef" は、関数内のトップインデントレベルでしか使えま
    # せん (他の場所で使えるようにするのは大したことではありません。
    # もしいい考えがあったら提案してください)。
    #
    # インデクスには、 "int" 型を使います。この型は C の int 型に
    # 対応しています。 ("unsigned int" のような) 他の型も使えます。
    # アレイのインデクスとして適切な型を使いたい純粋主義の人は、
    # "Py_ssize_t" を使ってもかまいません。
    cdef int vmax = f.shape[0]
    cdef int wmax = f.shape[1]
    cdef int smax = g.shape[0]
    cdef int tmax = g.shape[1]
    cdef int smid = smax // 2
    cdef int tmid = tmax // 2
    cdef int xmax = vmax + 2*smid
    cdef int ymax = wmax + 2*tmid
    # 出力画像の確保
    cdef np.ndarray[DTYPE_t, ndim=2] h = np.zeros([xmax, ymax], dtype=f.dtype)
    cdef int x, y, s, t                     # changed
    cdef unsigned int v, w      # changed
    # 変数全てについて型を定義するのがとても大事です。型を定義し忘
    # れても何の警告もでませんが、(変数が暗黙のうちに Python オブ
    # ジェクトに片付けされるので) コードは極端に遅くなります。
    cdef int s_from, s_to, t_from, t_to
    # 変数 tmps に対しては、アレイに保存されいているのと同じデー
    # タ型を使いたいので、上で定義した "DTYPE_t" を使います。
    # 注! この操作には、重大な副作用があります。 "tmps[x]" がデータ
    # 型の定義域をオーバフローすると、 Python のように例外が送出さ
    # れるのではなく、 C のときと同様に単なる桁落ち (wrap around)
    # を起こします。
    cdef np.ndarray[DTYPE_t, ndim=1] tmps = np.zeros(xmax, dtype=f.dtype)
    # コンボリューションの演算
    with nogil:
      for x in prange(xmax, schedule='static'):
          for y in range(ymax):
              # (x,y) におけるピクセル値 h を計算。
              # フィルタ g の各ピクセル (s, t) に対するコンポーネン
              # トを加算
              s_from = max(smid - x, -smid)
              s_to = min((xmax - x) - smid, smid + 1)
              t_from = max(tmid - y, -tmid)
              t_to = min((ymax - y) - tmid, tmid + 1)
              tmps[x] = 0
              for s in range(s_from, s_to):
                  for t in range(t_from, t_to):
                      v = <unsigned int>(x - smid + s)       # changed
                      w = <unsigned int>(y - tmid + t)       # changed
                      tmps[x] += g[<unsigned int>(smid - s), <unsigned int>(tmid - t)] * f[v, w] # changed
              h[x, y] = tmps[x]
    return h

@cython.boundscheck(False) # 関数全体で境界チェックを無効化
def naive_convolve(np.ndarray[DTYPE_t, ndim=2] f, np.ndarray[DTYPE_t, ndim=2] g):
    # f は画像で、 (v, w) でインデクスする
    # g はフィルタカーネルで、 (s, t) でインデクスする
    #   ディメンジョンは奇数でなくてはならない
    # h は出力画像で、 (x, y) でインデクスする
    #   クロップしない
    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")
    # smid と tmid は中心ピクセルからエッジまでのピクセル数、
    # つまり 5x5 のフィルタならどちらも 2
    #
    # 出力画像のサイズは入力画像の両端に smid と tmid を足した
    # 値となる
    assert f.dtype == DTYPE and g.dtype == DTYPE
    # "cdef" キーワードは、関数の中で変数の型を宣言するのにも使い
    # ます。 "cdef" は、関数内のトップインデントレベルでしか使えま
    # せん (他の場所で使えるようにするのは大したことではありません。
    # もしいい考えがあったら提案してください)。
    #
    # インデクスには、 "int" 型を使います。この型は C の int 型に
    # 対応しています。 ("unsigned int" のような) 他の型も使えます。
    # アレイのインデクスとして適切な型を使いたい純粋主義の人は、
    # "Py_ssize_t" を使ってもかまいません。
    cdef int vmax = f.shape[0]
    cdef int wmax = f.shape[1]
    cdef int smax = g.shape[0]
    cdef int tmax = g.shape[1]
    cdef int smid = smax // 2
    cdef int tmid = tmax // 2
    cdef int xmax = vmax + 2*smid
    cdef int ymax = wmax + 2*tmid
    # 出力画像の確保
    cdef np.ndarray[DTYPE_t, ndim=2] h = np.zeros([xmax, ymax], dtype=f.dtype)
    cdef int s, t                     # changed
    cdef unsigned int x, y, v, w      # changed
    # 変数全てについて型を定義するのがとても大事です。型を定義し忘
    # れても何の警告もでませんが、(変数が暗黙のうちに Python オブ
    # ジェクトに片付けされるので) コードは極端に遅くなります。
    cdef int s_from, s_to, t_from, t_to
    # 変数 value に対しては、アレイに保存されいているのと同じデー
    # タ型を使いたいので、上で定義した "DTYPE_t" を使います。
    # 注! この操作には、重大な副作用があります。 "value" がデータ
    # 型の定義域をオーバフローすると、 Python のように例外が送出さ
    # れるのではなく、 C のときと同様に単なる桁落ち (wrap around)
    # を起こします。
    cdef DTYPE_t value
    # コンボリューションの演算
    for x in range(xmax):
        for y in range(ymax):
            # (x,y) におけるピクセル値 h を計算。
            # フィルタ g の各ピクセル (s, t) に対するコンポーネン
            # トを加算
            s_from = max(smid - <int>(x), -smid)
            s_to = min((xmax - <int>(x)) - smid, smid + 1)
            t_from = max(tmid - <int>(y), -tmid)
            t_to = min((ymax - int(y)) - tmid, tmid + 1)
            value = 0
            for s in range(s_from, s_to):
                for t in range(t_from, t_to):
                    v = <unsigned int>(x - smid + s)       # changed
                    w = <unsigned int>(y - tmid + t)       # changed
                    value += g[<unsigned int>(smid - s), <unsigned int>(tmid - t)] * f[v, w] # changed
            h[x, y] = value
    return h
