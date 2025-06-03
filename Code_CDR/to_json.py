import json
from tqdm import tqdm


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def read_cdr(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)): #each document
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs: #each triplet
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = cdr_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(cdr_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features


def list_equal(l1, l2):
    if len(l1) != len(l2):
        return False
    for i in range(len(l1)):
        check = False
        for j in range(len(l1)):
            if l1[i] == l2[j]:
                check = True
        if not check:
            return False
    return True

def check_add(arr, item):
    add = True
    for i in arr:
        if len(i) != len(item):
            continue
        if list_equal(i, item):
            add = False
            break
    if add:
        arr.append(item)


def idx_of(arr, item):
    for index, i in enumerate(arr):
        if len(i) != len(item):
            continue
        if list_equal(i, item):
            return index
    raise Exception("No way.")





def transform(file_in, file_out):
    output = []
    pmids = set()

    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)): #each document
            per_doc = {}
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid in pmids:
                continue
            pmids.add(pmid)
            text = line[1]
            prs = chunks(line[2:], 17)

            vertexs = list()#each entity
            labels = []
            title = pmid
            sents = [t.split(' ') for t in text.split('|')]
            for p in prs:  # each triplet
                vertex1 = list() # first entity in the triplet, list of mentions
                mention_names1 = p[6].split('|')
                type1 = "CHEM" if p[7] == 'Chemical' else "DISE" if p[7] == 'Disease' else None
                assert type1 is not None
                sent_ids1 = p[10].split(':')
                positions_start1 = p[8].split(':')
                positions_end1 = p[9].split(':')
                for i in range(len(mention_names1)):
                    vertex1.append({'pos':[int(positions_start1[i]), int(positions_end1[i])],
                                    'type': type1,
                                    'sent_id': int(sent_ids1[i]),
                                    'name': mention_names1[i]
                                    })

                vertex2 = list() # first entity in the triplet, list of mentions
                mention_names2 = p[12].split('|')
                type2 = "CHEM" if p[13] == 'Chemical' else "DISE" if p[13] == 'Disease' else None
                assert type2 is not None
                sent_ids2 = p[16].split(':')
                positions_start2 = p[14].split(':')
                positions_end2 = p[15].split(':')
                for i in range(len(mention_names2)):
                    vertex2.append({'pos':[int(positions_start2[i]), int(positions_end2[i])],
                                    'type': type2,
                                    'sent_id': int(sent_ids2[i]),
                                    'name': mention_names2[i]
                                    })
                check_add(vertexs, vertex1)
                check_add(vertexs, vertex2)
                id1 = idx_of(vertexs, vertex1)
                id2 = idx_of(vertexs, vertex2)
                # vertexs.append(vertex1)
                # vertexs.append(vertex2)
                # id1 = len(vertexs) - 2
                # id2 = len(vertexs) - 1

                if p[0] != 'not_include':
                    r = p[0].split(':')[1]
                    if p[1] == 'L2R':
                        h = id1
                        t = id2
                    elif p[1] == 'R2L':
                        h = id2
                        t = id1
                    else:
                        raise Exception('Not provided!')
                    labels.append({'r': r, 'h': h, 't': t, 'evidence': [None]})



            per_doc['vertexSet']=vertexs
            per_doc['labels'] = labels
            per_doc['title'] = title
            per_doc['sents'] = sents
            output.append(per_doc)



    with open(file_out, "w") as final:
        json.dump(output, final)






if __name__ == '__main__':
    transform('Data/CDR/Original/Source/test_filter.data', 'Data/CDR/Original/test.json')
    transform('Data/CDR/Original/Source/dev_filter.data', 'Data/CDR/Original/dev.json')
    transform('Data/CDR/Original/Source/train_filter.data', 'Data/CDR/Original/train_annotated.json')

    # line = "14596845	A diet promoting sugar dependency causes behavioral cross - sensitization to a low dose of amphetamine .|Previous research in this laboratory has shown that a diet of intermittent excessive sugar consumption produces a state with neurochemical and behavioral similarities to drug dependency .|The present study examined whether female rats on various regimens of sugar access would show behavioral cross - sensitization to a low dose of amphetamine .|After a 30 - min baseline measure of locomotor activity ( day 0 ) , animals were maintained on a cyclic diet of 12 - h deprivation followed by 12 - h access to 10 % sucrose solution and chow pellets ( 12 h access starting 4 h after onset of the dark period ) for 21 days .|Locomotor activity was measured again for 30 min at the beginning of days 1 and 21 of sugar access .|Beginning on day 22 , all rats were maintained on ad libitum chow .|Nine days later locomotor activity was measured in response to a single low dose of amphetamine ( 0 . 5 mg / kg ) .|The animals that had experienced cyclic sucrose and chow were hyperactive in response to amphetamine compared with four control groups ( ad libitum 10 % sucrose and chow followed by amphetamine injection , cyclic chow followed by amphetamine injection , ad libitum chow with amphetamine , or cyclic 10 % sucrose and chow with a saline injection ) .|These results suggest that a diet comprised of alternating deprivation and access to a sugar solution and chow produces bingeing on sugar that leads to a long lasting state of increased sensitivity to amphetamine , possibly due to a lasting alteration in the dopamine system .	1:CID:2	R2L	NON-CROSS	202-203	198-199	D000661	amphetamine|amphetamine|amphetamine|amphetamine|amphetamine|amphetamine|amphetamine|amphetamine	Chemical	15:68:178:202:218:225:232:280	16:69:179:203:219:226:233:281	0:2:6:7:7:7:7:8	D006948	behavioral cross - sensitization|behavioral cross - sensitization|hyperactive	Disease	6:59:198	10:63:199	0:2:7	1:CID:2	R2L	NON-CROSS	198-199	194-195	D013395	sucrose|sucrose|sucrose|sucrose	Chemical	106:194:213:238	107:195:214:239	3:7:7:7	D006948	behavioral cross - sensitization|behavioral cross - sensitization|hyperactive	Disease	6:59:198	10:63:199	0:2:7	1:NR:2	R2L	NON-CROSS	15-16	3-5	D000661	amphetamine|amphetamine|amphetamine|amphetamine|amphetamine|amphetamine|amphetamine|amphetamine	Chemical	15:68:178:202:218:225:232:280	16:69:179:203:219:226:233:281	0:2:6:7:7:7:7:8	D019966	sugar dependency|drug dependency	Disease	3:41	5:43	0:1	1:NR:2	R2L	CROSS	106-107	41-43	D013395	sucrose|sucrose|sucrose|sucrose	Chemical	106:194:213:238	107:195:214:239	3:7:7:7	D019966	sugar dependency|drug dependency	Disease	3:41	5:43	0:1	1:NR:2	R2L	CROSS	290-291	41-43	D004298	dopamine	Chemical	290	291	8	D019966	sugar dependency|drug dependency	Disease	3:41	5:43	0:1	1:NR:2	R2L	CROSS	290-291	198-199	D004298	dopamine	Chemical	290	291	8	D006948	behavioral cross - sensitization|behavioral cross - sensitization|hyperactive	Disease	6:59:198	10:63:199	0:2:7"
    #
    #
    #
    # line = line.rstrip().split('\t')
    # text = line[1]
    # sents = [t.split(' ') for t in text.split('|')]
    # print(sents[2])
    # print(sents[6])
    # # print(text)
    # prs = chunks(line[2:], 17)
    # for i in prs:
    #     print(i)
    # test = prs[0]
    # print(test)
    # print(test[14])
    # print(test[15])

