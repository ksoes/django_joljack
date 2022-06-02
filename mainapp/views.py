#-*- coding: utf-8 -*-
from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.utils import timezone

from functionfile import convert_pdf_to_text, convert_hwp_to_text,  convert_txt_to_text, convert_docx_to_text, convert_link_to_text, clean_sentence, \
    fake_text_find, get_recommendations, make_table, check_table, put_Rapi, put_Bapi, put_Cword, put_FCword, check_presence_data, \
    take_Rapi, take_Bapi, take_Cword, take_FCword
from .forms import UploadForm, EvaluateForm
from .models import Result, EvaluateTable

from konlpy.tag import Mecab
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # tf-idf
from sklearn.metrics.pairwise import linear_kernel # tf-idf

from urllib.parse import unquote_plus
from urllib import parse

# Create your views here.


def index(request):
    # 첫 페이지
    return render(request, 'mainapp/start.html')


def information(request):
    # 설명 페이지
    return render(request, 'mainapp/information.html')


def result(request):
    # 결과 페이지
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)

        # if request.FILES['file'] :
        if form['file'].value() is not None:  # 입력이 file 인 경우
            if form.is_valid():
                f = request.FILES['file']  # 파일불러옴
                fs = FileSystemStorage()  # 파라미터 없으면 저장되는 경로: settings.py의 MEDIA_ROOT
                filename = fs.save(f, f)  # 파일을 경로에 저장
                url = fs.url(filename)  # 이름으로 참조된 파일의 내용에 액세스 할 수 있는 url을 반환
                file_path = url.split('.')  # .을 기준으로 <경로,파일명> 과 <파일형식>을 구분
                file_path.insert(-1, '.')  # 나누는 기준으로 없어져버린 . 추가 (마지막 원소인 <형식> 앞에 .을 추가)
                path = file_path[0][:7]  # 나눠진 <경로,파일> 에서 <경로>(/media/) 부분을 저장
                file_path[0] = path+parse.unquote(file_path[0][7:])  # parse.unquote()로 인코딩된 문자열을 디코딩해서 다시 저장
                url = ''.join(file_path)  # 구분한 경로,파일 과 형식을 합쳐 파일경로를 만든다
                print("6. ", url)

                if url.split('.')[-1] == 'pdf':
                    v = convert_pdf_to_text(url)
                elif url.split('.')[-1] == 'hwp':
                    v = convert_hwp_to_text(url)
                elif url.split('.')[-1] == 'docx':
                    v = convert_docx_to_text(url)
                elif url.split('.')[-1] == 'txt':
                    v = convert_txt_to_text(url)

        else:
            if form.is_valid():
                url1 = unquote_plus(form['url'].value())
                v = convert_link_to_text(url1)

        body_text = v
        sentence = clean_sentence(v)
        # ------------------------------------------------------------ 1
        # Mecab으로 명사 추출
        mecab = Mecab()
        final = mecab.nouns(sentence)  # mecab으로 명사 추출

        # ------------------------------------------------------------ 2
        # 추출된 명사들 중 빈도수가 3이상인 명사들만 보관
        word_list = pd.Series(final)  # 시리즈로 변경
        result = word_list.value_counts()  # 개수 구하기
        result_df = pd.DataFrame(result)  # 데이터프레임으로 변경
        result_df.columns = ['빈도수']  # 컬럼명 '빈도수'로 수정

        # 빈도수 3 이상인 단어만 데이터프레임으로 따로 보관
        freq = result_df[(result_df['빈도수'] >= 3)]
        count = pd.DataFrame(freq.index, columns=['word'])  # freq(df) count(df)로 새로 생성
        # ------------------------------------------------------------ 3

        # ------------------------------------------------------------ 4
        # """ # 2. 전체주석 없애버림 598~643
        checktable = 0  # 테이블 유무 확인변수
        checktable = check_table()
        if checktable == 0:  # 빈 테이블이 없다면
            print('테이블 없으므로 생성')
            make_table()
            print(' - 테이블 생성 완료')
        else:
            print('*이미 테이블이 생성되어있으므로 테이블 생성 건너뜀*')
        # ------------------------------------------------------------ 5

        sqlapi, sqlban, sqlco, sqlfco = check_presence_data()

        if sqlapi == 0:
            print('api테이블 데이터 없으므로 저장')
            total_df = put_Rapi()
            print(' - Rapi(등록)완료')
        else:
            print('*api테이블에 데이터가 있으므로 저장 건너뜀*')
            total_df = take_Rapi()  # mysql에서 데이터가져와서 df에 저장

        if sqlban == 0:
            print('ban테이블에 데이터 없으므로 저장')
            bantotal_df1 = put_Bapi()
            print('- Bapi(금지)완료')
        else:
            print('*ban테이블에 데이터가 있으므로 저장 건너뜀*')
            bantotal_df1 = take_Bapi()  # mysql에서 데이터가져와서 df에 저장

        if sqlco == 0:
            print('coword테이블에 데이터 없으므로 저장')
            df = put_Cword()
            print('- Cword(동출빈)완료')
        else:
            print('*coword테이블에 데이터가 있으므로 저장 건너뜀*')
            df = take_Cword()  # mysql에서 데이터가져와서 df에 저장

        if sqlfco == 0:
            print('fcoword테이블에 데이터 없으므로 저장')
            df2 = put_FCword()
            print('- FCword(fake동출빈)완료')
        else:
            print('*fcoword테이블에 데이터가 있으므로 저장 건너뜀*')
            df2 = take_FCword()  # mysql에서 데이터가져와서 df에 저장


        # 등록상품
        total_prdlst = np.array(total_df['PRDLST_NM'].tolist())
        total_bssh = np.array(total_df['BSSH_NM'].tolist())

        # 위해상품
        ban_prdt = np.array(bantotal_df1['PRDT_NM'].tolist())
        ban_mufc = np.array(bantotal_df1['MUFC_NM'].tolist())

        # final1 중복값 제거
        final1 = final
        final1 = set(final1)
        sample = list(final1)

        # db에 일치하는 결과가 있는 단어만 따로 빼서 저장할 리스트
        sample_list_total = []
        sample_list_ban = []

        for i in range(len(sample)):
            if sample[i] in total_prdlst:
                if sample[i] not in sample_list_total:
                    sample_list_total.append(sample[i])
            if sample[i] in total_bssh:
                if sample[i] not in sample_list_total:
                    sample_list_total.append(sample[i])
            if sample[i] in ban_prdt:
                if sample[i] not in sample_list_ban:
                    sample_list_ban.append(sample[i])
            if sample[i] in ban_mufc:
                if sample[i] not in sample_list_ban:
                    sample_list_ban.append(sample[i])

        print('등록된 제품명 : {0}'.format(sample_list_total))
        print('등록된 위해제품명 : {0}'.format(sample_list_ban))

        # ------------------------------------------------------------ 7
        # 동시출현 쌍 만들기.. (한문서)
        co_word = {}
        for i in range(len(final)):
            for j in range(len(final)):
                a = final[i]
                b = final[j]
                if a == b: continue  # 둘이 같은 단어인 경우는 세지 않음
                if a > b: a, b = b, a  # a, b와 b, a 가 다르게 세어지는 것을 막기 위해 순서 고정
                co_word[a, b] = co_word.get((a, b), 0) + 1 / 2  # 실제로 센다

        co_word_list = list(co_word)  # co_word 사전에서 key부분 (단어쌍 부분)만 추출해서 리스트로 변환
        co_freq_list = list(co_word.values())  # co_word 사전에서 value부분 (나온 횟수)만 추출해서 리스트로 변환

        # co_word_list 리스트를 df로 변환
        co_word_df = pd.DataFrame(co_word_list, columns=['word1', 'word2'])

        # df에 새 컬럼 'freq'를 추가하고, 값으로 co_freq_list (나온 횟수) 집어넣기
        co_word_df['freq'] = co_freq_list

        # 한 문서 내의 단어 쌍 df
        co_word_df

        # 신뢰도 높음 동출빈과 비교해 일치하는 것을 따로 저장
        df_merge = co_word_df.merge(df)
        # print("\n신뢰도 높음과 일치\n",df_merge)

        # 신뢰도 낮음 동출빈과 비교해 일치하는 것을 따로 저장
        df2_merge = co_word_df.merge(df2)
        # print("\n신뢰도 낮음과 일치\n",df2_merge)

        # -------------------------------------------------------- 8
        # 기사의 유사도를 구하는 부분

        # 입력한 기사의 본문 불러오기
        tf_article = sentence

        # 기사자료.csv 불러오고 중복제거
        data = pd.read_csv('./etc_file/article_data.csv', low_memory=False)  # csv 읽어오기
        data_df = data[['article', 'code']]  # data(df) 열이름 변경
        data_df.drop_duplicates()  # 중복제거

        # 입력한 기사를 df에 추가
        data_insert = {'article': tf_article, 'code': '불명'}  # 기사본문(1개)
        data_df = data_df.append(data_insert, ignore_index=True)  # data_df(csv읽어온df)에 추가

        indices = pd.Series(data_df.index, index=data_df['article'])  # 합친 기사본문 가지고 판다스 시리즈 생성

        tfidf = TfidfVectorizer()

        # title에 대해서 tf-idf 수행
        tfidf_matrix = tfidf.fit_transform(data_df['article'])

        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # get_recommendations 함수의 결과를 받아올 df 생성
        get_df = pd.DataFrame(get_recommendations(tf_article, cosine_sim, indices, data_df))

        # print(get_df)
        code_number = get_df.index[get_df['code'] == "높음"]  # 유사도가 높은 기사의 인덱스 추출

        article_body = []
        for i in code_number:  # code_number에 있는 '높음'기사 번호로
            article_body.append(data_df.loc[i]['article'])  # data_df(기사있는 df)에서 기사 본문만 따로 저장하기

        article_body_length = len(article_body)  # 유사도 '높음' 기사 개수

        value_series = get_df['code'].value_counts()

        # 보통도 있을 수 있는데 보통은 계산에 포함X
        try:
            high = value_series['높음']
        except KeyError:
            high = 0
        try:
            low = value_series['낮음']
        except KeyError:
            low = 0
        print('high :', high, ", low :", low)

        tf_num = (high - low) * 6

        # -------------------------------------------------------- 9
        reliability = 0  # 신뢰도 넣을 변수
        fake_count = fake_text_find(v)  # 함수 조회로 카운트 받기

        if len(sample_list_total) == 0 and len(sample_list_ban) == 0:
            print("신뢰도 측정 대상 외의 기사입니다.")
            return render(request, 'mainapp/other.html')

        else:
            # 신뢰도 총 70점
            if sample_list_total and not sample_list_ban:
                reliability += 50
                print("등록점수 :", reliability)
            elif not sample_list_total and sample_list_ban:
                reliability += 20
                print("등록점수 :", reliability)
            else:  # 둘 다 겹칠 때
                reliability += 20
                print("등록점수 :", reliability)

            # 동시출현단어 빈도수 비교 (최대 점수 30점)
            if len(df_merge.index) >= len(df2_merge.index):
                num = int(len(df2_merge.index) / len(df_merge.index) * 20)
                num = 20 - num
            elif len(df_merge.index) < len(df2_merge.index):
                num = int(len(df_merge.index) / len(df2_merge.index) * 20)

            # num이 30이 넘거나 음수인 경우 값 조정
            if num > 20:
                num = 20
            elif num < 0:
                num = 0
            reliability += num
            print("동출빈점수 :", num)

            # tf-idf 유사도 점수적용
            reliability += tf_num
            print("tf-idf점수 :", tf_num)

            if fake_count != 0:  # 허위과대광고 문구가 본문에 있을경우
                reliability -= 1 * fake_count
                print("허위과대광고문구가 {0}개 있으므로 {1}점 감소".format(fake_count, 1 * fake_count))

            # result_page에 출력할 본문 정리
            body_text = body_text.replace("\n", '')
            body_text = body_text.replace("\x0c", '')
            list_bd = body_text.split(sep='. ')  # ". "으로 한 문장씩 나눠서 리스트로 정리
            list_bd = [item + '.' for item in list_bd]  # list로 나눠진 문장끝에 . 추가
            body_text = list_bd

            # Result 테이블에 실행결과 저장
            if body_text:
                if form['file'].value():
                    u = url  # 파일경로 저장
                elif form['url'].value():
                    u = url1  # 기사 url 저장
                else:
                    print('저장Error: 실행된 결과가 없습니다.')
                b = body_text
                t = timezone.now()
                try:
                    Result(uploaded=u, body=b, create_date=t).save()
                except:
                    print("db에 저장 중 문제가 발생했습니다.")

    elif request.method == 'GET':
        form = UploadForm(prefix='upload')
        return render(request, 'mainapp/add.html', {'form': form})

    context = {'body_text': body_text, 'percentage': reliability, 'number': len(body_text), 'sample_list_total': sample_list_total, 'sample_list_ban': sample_list_ban, 'article_body': article_body, 'article_body_length': article_body_length}

    return render(request, 'mainapp/result.html', context)


def evaluate(request):
    if request.method == 'POST':
        form = EvaluateForm(request.POST)
        if form.is_valid():
            option = form['option'].value()
            text = form['eval_text'].value()
            result = Result.objects.last()
            print(option, text, result, result.id)
            try:
                EvaluateTable(option_gb=option, eval_text=text, result_id=result.id).save()
            except:
                print("db 저장 중 문제발생")

        else:
            print("4. Error :", form.errors)

    elif request.method == 'GET':
        form = EvaluateForm()
        return render(request, 'mainapp/write.html', {'form': form})

    context = {'eval_option': option, 'eval_text': text}
    return render(request, 'mainapp/evaluate.html', context)

