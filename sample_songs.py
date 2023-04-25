from extractors.easy_songwriting import SampleSong
from mir.music_base import get_scale_and_suffix

song_tongnian_8=SampleSong('tongnian',8,
                           '[+3]00355--3|66760665||1-111616|5-------|'
                           '[+3]00355--3|66760665||1-111661|2-------|',
                           'C:maj---|A:min---||F:maj---|G:maj---'
                           'C:maj---|A:min---||F:maj---|G:maj---',text_beat_division=2,bpm=160)

song_chonger_8=SampleSong('chonger',8,
                          '[-6]3-334-5-|3---2---||1-112-3-|3---7---|'
                          '[-6]6-3-2---|6-3-2---||6-3-2--1|1-------|',
                          'C:maj---|G:maj---||A:min---|E:min---|'
                          'F:maj-G:maj-|F:maj-G:maj-||F:maj-G:maj-|A:min---|',text_beat_division=2,bpm=160)

song_pengyou_8=SampleSong('pengyou',8,
                          '[+2]5-5-5-6-|5---6-7-||    1-6-6-5-|3---3-2-|'
                          '[+1]1---1-6-|5---3-2-||[-6]1---6-1-|2---3-5-|'
                          '[+2]5-5-5-6-|5---6-7-||    1-6-6-5-|3---3-2-|'
                          '[+1]1---1-6-|5---3-2-||[-6]1---6-1-|1-------|',
                          'C:maj-|G:maj-||A:min-|E:min-'
                          'F:maj-|C:maj-||D:min-|G:maj-'
                          'C:maj-|G:maj-||A:min-|E:min-'
                          'F:maj-|C:maj-||D:min,G:maj|C:maj-',text_beat_division=4,bpm=160)

song_pfzl_8=SampleSong('pfzl',8,'[+2]1-71-56-|6---5-43||    --3-3-32|--5-6-7-|'
                                   '    1-71-56-|6---6-71||[+3]--1-1-12|--5-6-7-|'
                                   '[+7]1-71-3--|6--5--4-||[+6]3---3-2-|----6-7-|'
                                   '    1---1-21|---171--||    2---1-71|--------|',
                                   'A:min-|F:maj-||C:maj-|G:maj-|'
                                   'A:min-|F:maj-||C:maj-|G:maj-|'
                                   'A:min-|F:maj-||C:maj-|G:maj-|'
                                   'A:min-|F:maj-||G:maj-|C:maj-|',text_beat_division=4,bpm=160)

song_juejiang_8=SampleSong('juejiang',8,
                           '[-7]1-----11|71-2--71||    ------11|71-2--31|'
                           '[-5]------11|1-51-3--||    4-3-2-1-|3--22---|'
                           '[-7]1-----11|71-2--71||    ------11|71-2--31|'
                           '[-5]------11|1-51-3--||[+1]4-3-2-1-|6--55---|',
                           'C:maj-|G:maj-||A:min-|E:min-'
                           'F:maj-|C:maj-||D:min-|G:maj-'
                           'C:maj-|G:maj-||A:min-|E:min-'
                           'F:maj-|C:maj-||D:min-|G:maj-',text_beat_division=4,bpm=160)

littlestar_8=SampleSong('littlestar',8,
                          '[+1]1---1---5---5---|6---6---5---5---||'
                          '    4---4---3---3---|2---2--31-------||'
                          '[+1]1---1---5---5---|6---6---5---5---||'
                          '    4---4---3---3---|2---2--31-------||',
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||'
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||',text_beat_division=4,bpm=160)

littlestar_8_oct_high=SampleSong('littlestar+8',8,
                          '[+7]1---1---5---5---|6---6---5---5---||'
                          '    4---4---3---3---|2---2--31-------||'
                          '[+7]1---1---5---5---|6---6---5---5---||'
                          '    4---4---3---3---|2---2--31-------||',
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||'
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||',text_beat_division=4,bpm=160)
littlestar_8_oct_low=SampleSong('littlestar-8',8,
                          '[-1]1---1---5---5---|6---6---5---5---||'
                          '    4---4---3---3---|2---2--31-------||'
                          '[-1]1---1---5---5---|6---6---5---5---||'
                          '    4---4---3---3---|2---2--31-------||',
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||'
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||',text_beat_division=4,bpm=160)
littlestar_fast_8=SampleSong('littlestar(fast)',8,
                          '[-7]21717171|65#45#45#45||[+4]#56172176|65321765|'
                          '[+3]54217654|43176543||[-7]2-6-5-7-|1-------'
                          '[-7]21717171|65#45#45#45||[+4]#56172176|65321765|'
                          '[+3]54217654|43176543||[-7]2-6-5-7-|1-------',
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||'
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||',text_beat_division=4,bpm=160)
test_fast_8=SampleSong('Test',8,
                          '1234543212345432|2345654323456543'
                          '1234543212345432|3456765434567654'
                          '1234543212345432|2345654323456543'
                          '1234543212345432|4567[+2]176545671765',
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||'
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||',text_beat_division=4,bpm=160)
test2_fast_8=SampleSong('Test2',8,
                          '1234543212345432|1234543212345432'
                          '2543232125654543|2543232125654543'
                          '[-5]52325232[+1]15651565|[-5]52325232[+1]15651565'
                          '3535454213132325|3535454213132325',
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||'
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||',text_beat_division=4,bpm=160)
test3_fast_8=SampleSong('Test2',8,
                          '1234543212345432|1234543212345432'
                          '1234543212345432|[-5]52325232[+1]15651565'
                          '[-5]52325232[+1]15651565|[-5]52325232[+1]15651565'
                          '3535454213132325|3535454213132325',
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||'
                          'C:maj---|F:maj-C:maj-||G:maj-C:maj-|G:maj-C:maj-||',text_beat_division=4,bpm=160)

xiaocheng_8=SampleSong('xiaocheng',8,
                       '[+2]3--56561|5-------||    5--63235|2-------||'
                       '[+2]2--35-61|6-616-5-||[+1]5-523-2-|1-------||',
                       'C:maj---|C:maj---||A:min---|D:min---||'
                       'G:maj-C:maj-|A:min---||G:maj---|C:maj---',text_beat_division=2,bpm=160)

doraemon_8=SampleSong('doraemon',8,
                       '[-5]5--11--[-7]3|6--35---||    5--65--3|4--32---||'
                       '    7--22--4|[+1]7--76--5||[-6]4--44--3|6--7--1-||'
                       '2-------|--------||00000000|00000000||'
                       '00000000|00000000||00000000|00000000||',
                       'C:maj---|C:maj-G:maj-||G:maj---|G:maj---||'
                       'G:maj---|N---||N---|N---',text_beat_division=4,bpm=160)

hanon_a_8=SampleSong('hanon_a',8,
                     '[+1]13456543|24567654||[+3]35671765|46712176||'
                     '[+5]57123217|61234321||[+7]72345432|13456543||',
                     '',text_beat_division=2,bpm=160)
hanon_b_8=SampleSong('hanon_b',8,
                     '[+1]16564636|27675747||[+3]31716151|42127262||'
                     '[+5]53231373|64342414||[+7]75453525|16564636||',
                     '',text_beat_division=2,bpm=160)

love_gate_8=SampleSong('LoveAtTheGate',8,
                       '[+1]1231|5---|[-4]6716|3---|5671|5---|4-[+1]34|2---|',
                       'C:maj---|G:maj---||A:min---|E:min---||'
                       'F:maj---|C:maj---||D:min---|G:maj---',text_beat_division=1,bpm=160)

downwards=SampleSong('Downwards',8,
                       '5432|1---|6543|2---|7654|3---|[+2]1765|4---|',
                       'C:maj---|G:maj---||A:min---|E:min---||'
                       'F:maj---|C:maj---||D:min---|G:maj---',text_beat_division=1,bpm=160)

upwards=SampleSong('Upwards',8,
                       '1234|5---|2345|6---|3456|7---|[+2]4567|1---|',
                       'C:maj---|G:maj---||A:min---|E:min---||'
                       'F:maj---|C:maj---||D:min---|G:maj---',text_beat_division=1,bpm=160)

upwards2=SampleSong('Upwards2',8,
                       '1234|5---|23#45|6---|3#4#56|7---|[+2]456b7|1---|',
                       'C:maj---|G:maj---||A:min---|E:min---||'
                       'F:maj---|C:maj---||D:min---|G:maj---',text_beat_division=1,bpm=160)

downwards2=SampleSong('Downwards2',8,
                       '5432|1---|65#43|2---|76#5#4|3---|[+2]1b765|4---|',
                       'C:maj---|G:maj---||A:min---|E:min---||'
                       'F:maj---|C:maj---||D:min---|G:maj---',text_beat_division=1,bpm=160)

joy=SampleSong('Joy',8,
                       '3-3-4-5-|5-4-3-2-|1-1-2-3-|3--22---|3-3-4-5-|5-4-3-2-|1-1-2-3-|2--11---|',
                       'C:maj---|G:maj---||C:maj---|G:maj---||'
                       'C:maj---|G:maj---||C:maj---|G:maj-C:maj-',text_beat_division=2,bpm=160)

tiger=SampleSong('2Tiger',8,
                       '1-2-3-1-|1-2-3-1-|3-4-5---|3-4-5---|56543-1-|56543-1-|[-5]2-5-1---|2-5-1---|',
                       'C:maj---|C:maj---||C:maj---|C:maj---||'
                       'C:maj---|C:maj---||G:maj-C:maj-|G:maj-C:maj-',text_beat_division=2,bpm=160)

mary=SampleSong('Mary',8,
                       '3--21-2-|3-3-3---|2-2-2---|3-5-5---|3--21-2-|3-3-3-3-|2-2-3-2-|1-------|',
                       'C:maj---|C:maj---||G:maj---|C:maj---||'
                       'C:maj---|C:maj---||G:maj---|C:maj---',text_beat_division=2,bpm=160)

jingle_bells=SampleSong('Jingle Bells',8,
                        '[-5]5-3-2-1-|5-----55|5-3-2-1-|6-----00|6-4-3-2-|7-----00|[+1]5-5-4-2-|3-------|',
                       'C:maj---|C:maj---||C:maj---|F:maj---||'
                       'D:min---|G:maj---||G:7---|C:maj---',text_beat_division=2,bpm=160)

#xiaocheng2_8=SampleSong('xiaocheng2',8,
#                       '[-5]16123---|23615---||6123[+2]5316|5-------||'
#                       '[+2]1-165-56|5-532---||[+1]5-563-2-|1-------||',
#                       'A:min---|D:min-G:maj-||C:maj---|G:maj---||'
#                       'A:min---|E:min-D:min-||G:maj---|C:maj---',text_beat_division=2,bpm=160)

song_8_list=[song_tongnian_8,song_chonger_8,song_pfzl_8,song_pengyou_8,song_juejiang_8,xiaocheng_8,littlestar_8,littlestar_fast_8,doraemon_8,hanon_a_8,hanon_b_8,love_gate_8]

song_8_list_new=[song_tongnian_8,song_chonger_8,xiaocheng_8,littlestar_8,littlestar_fast_8,mary,joy,tiger,jingle_bells,littlestar_8_oct_high,littlestar_8_oct_low]

song_upwards=[upwards,downwards,upwards2,downwards2]


song_pfzl=SampleSong('平凡之路',2,'[+2]1-71-56-|6---5-43||--3-3-32|--------','A:min-|F:maj-||C:maj-|G:maj-',text_beat_division=4)

song_tongnian=SampleSong('童年',2,'[+3]00355--3|66760665||1-111616|5-------','C:maj-|A:min-||F:maj-|G:maj-',text_beat_division=4)

song_timian=SampleSong('体面',2,'[-6]3123--15|5--5321-||1671--63|3--2351-','C:maj-|G:maj-||A:min-|E:min-',text_beat_division=4)

song_chonger=SampleSong('虫儿飞',2,'[-7]3-334-5-|3---2---||1-112-3-|3---7---','C:maj-|G:maj-||A:min-|E:min-',text_beat_division=4)

song_juejiang=SampleSong('倔强',2,'[-7]1-----11|71-2--71||------11|71-2--31|','C:maj-|G:maj-||A:min-|E:min-',text_beat_division=4)

song_pengyou=SampleSong('朋友',2,'[+2]5-5-5-6-|5---6-7-||1-6-6-5-|3---3-2-|','C:maj-|G:maj-||A:min-|E:min-',text_beat_division=4)

song_fast_canon=SampleSong('卡农(16分音符)',2,'5-345-34|5[-5]5671234||3-123-12|3[-3]3456712|','C:maj-|G:maj-||A:min-|E:min-',text_beat_division=4)

song_slow_canon=SampleSong('卡农(2分音符)',2,'[-7]3-------|2-------||1-------|7-------|','C:maj-|G:maj-||A:min-|E:min-',text_beat_division=4)

song_fast_star=SampleSong('小星星(16分音符)',2,'[-7]21717171|65#45#45#45||[+4]67172176|65321765|','C:maj-|C:maj/3-||F:maj-|C:maj/3-',text_beat_division=4)
song_slower_star=SampleSong('小星星(2分音符)',2,'1-------|5-------||6-------|5-------|','C:maj-|C:maj/3-||F:maj-|C:maj/3-',text_beat_division=4)
song_slow_star=SampleSong('小星星(4分音符)',2,'1---1---|5---5---||6---6---|5-------|','C:maj-|C:maj/3-||F:maj-|C:maj/3-',text_beat_division=4)

song_hongyan=SampleSong('鸿雁',2,'[-5]2-3-1-6-|5-------||[+2]5---6-1-|6-------|','C:maj-|--||E:min-|A:min,A:min/b7',text_beat_division=4)

song_xiaocheng=SampleSong('小城故事',2,'[+2]3--56561|5-------||5-563235|2-------','C:maj-|--||A:min-|D:min-',text_beat_division=4)

song_huluwa=SampleSong('葫芦娃',2,'1-1-3---|11-3----||6-6-656-|51-3-----|','C:maj-|--||A:min-|C:maj-',text_beat_division=4)

song_twotigers_1=SampleSong('TwoTigers1',2,'1-2-3-1-|1-2-3-1-','C:maj---|C:maj---',text_beat_division=2)
song_twotigers_2=SampleSong('TwoTigers2',2,'3-4-5---|3-4-5---','C:maj---|C:maj---',text_beat_division=2)

song_ministar_2=SampleSong('Ministar1',2,'1-1-5-5-|6-6-5-5-','C:maj---|C:maj---',text_beat_division=2)


song_speed_adjuster=[
    song_slower_star,
    SampleSong('小星星(无重音4分音符)',2,'1---3---|5---[+2]1---||6---1---|5-------|','C:maj-|C:maj/3-||F:maj-|C:maj/3-',text_beat_division=4),
    SampleSong('小星星(无重音8分音符)',2,'1-2-3-1-|5-3-5-[+2]1-||6-7-1-2-|1-6-5---|','C:maj-|C:maj/3-||F:maj-|C:maj/3-',text_beat_division=4),
    song_fast_star
]

song_2_list=[song_huluwa,song_tongnian,song_pengyou,song_chonger,song_timian,song_pfzl,song_juejiang,song_fast_canon,song_xiaocheng,song_hongyan,
               song_slow_star,song_slower_star,song_fast_star,song_twotigers_1,song_twotigers_2,song_ministar_2]
if __name__ == '__main__':
    for song in [xiaoxingxing_8_oct_low]:
        song.visualize()
