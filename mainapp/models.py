from django.db import models

# Create your models here.
class BanApidata(models.Model):
    prdt_nm = models.CharField(db_column='PRDT_NM', primary_key=True, max_length=250)  # Field name made lowercase.
    mufc_nm = models.CharField(db_column='MUFC_NM', max_length=150)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'ban_apidata'
        unique_together = (('prdt_nm', 'mufc_nm'),)


class CoOccurrenceWord(models.Model):
    word1 = models.CharField(primary_key=True, max_length=30)
    word2 = models.CharField(max_length=30)
    count = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'co_occurrence_word'
        unique_together = (('word1', 'word2'),)


class EvaluateTable(models.Model):
    option_gb = models.CharField(max_length=10, blank=True, null=True)
    eval_text = models.CharField(max_length=1500, blank=True, null=True)
    result = models.ForeignKey('Result', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'evaluate_table'


class ExApidata(models.Model):
    prdlst_nm = models.CharField(db_column='PRDLST_NM', primary_key=True, max_length=250)  # Field name made lowercase.
    bssh_nm = models.CharField(db_column='BSSH_NM', max_length=150)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'ex_apidata'
        unique_together = (('prdlst_nm', 'bssh_nm'),)


class FakeCoOccurrenceWord(models.Model):
    word1 = models.CharField(primary_key=True, max_length=30)
    word2 = models.CharField(max_length=30)
    count = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'fake_co_occurrence_word'
        unique_together = (('word1', 'word2'),)


class Result(models.Model):
    uploaded = models.CharField(max_length=500)
    body = models.CharField(max_length=10000)
    create_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'result'
