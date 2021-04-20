import pymysql

conn = pymysql.connect(host='localhost', user='JH', password='0019', db='tsst', charset='utf8')

with conn.cursor() as cursor:
    # sql = 'insert into testtable(pos,ppm) values("p777",234);'

    sql = 'insert into ss(humid,temp) values(%s,%s);'
    cnt = cursor.execute(sql, ('p8', 123))
    r = conn.commit()

    if r == 0:
        print("failed")

    else:
        print("Save Ok")
conn.close()

