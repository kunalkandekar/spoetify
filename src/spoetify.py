import sys, os
import collections, bisect
import socket
import traceback
import array 
import time, datetime
import urllib, urllib2
import threading
import Queue
import json
import string, math, random

import logging

def log(s):
    sys.stderr.write(s+"\n")

def remove_non_ascii(s):
    s = "".join(i for i in s if ord(i) < 128)
    return s.encode("utf-8")

def normalize_str(token):
    otoken = token = remove_non_ascii(token.lower())
    try:
        if len(token) < 1:
            return token
        chf = token[0]
        open_quote = False
        if chf == '\'' or chf == '"':
            che = token[-1]
            if len(token) > 2 and (che == '\'' or che == '"'):
                token = token[1:-1]
        if len(token) < 1:
            return token    
        if chf in '({[\\/':
            token = token[1:]
        if len(token) < 1:
            return token
        if token[-3:] == "...":
            token = token[:-3]
        if len(token) < 1:
            return token
        chl = token[-1]
        if chl in ')}]\\/,.:;-`!?': # or chl in punct_sets:
            token = token[:-1]
    except:
        handle_exc(sys.exc_info())
        logging.error("Error while normalizing token='"+otoken+"'")
        raise
    return token


def put_as_list(d, k, v):
    l = d.get(k, None)
    if not l:
        l = []
        d[k] = l
    l.append(v)

def put_as_set(d, k, v):
    l = d.get(k, None)
    if not l:
        l = set()
        d[k] = l
    l.add(v)

def ngrams_from_tokens(tokens, ngramsize, overlap):
    ngrams = set()
    ntokens = len(tokens)
    jump = ngramsize - overlap
    
    for i in xrange(0, ntokens, jump):
        upto_ngram = i + ngramsize
        if upto_ngram < ntokens:
            ngram = tokens[i:upto_ngram]
        else:
            ngram = tokens[i:]
        ngrams.add(tuple(ngram))
    ngrams.add(tuple(tokens[:jump]))
    ngrams.add(tuple(tokens[i - jump + 1:]))
    return ngrams
    
def ngrams_from_linewise_tokens(linewise_tokens, ngramsize, overlap):
    ngrams = set()
    for tokens in linewise_tokens:
        ntokens = len(tokens)
        jump = ngramsize - overlap
        if jump > ntokens:
            ngrams.add(tokens)
            continue
        for i in xrange(0, ntokens, jump):
            upto_ngram = i + ngramsize
            if upto_ngram < ntokens:
                ngram = tokens[i:upto_ngram]
            else:
                ngram = tokens[i:]
            ngrams.add(tuple(ngram))
        ngrams.add(tuple(tokens[:jump]))
        #finish off the tail end
        ngrams.add(tuple(tokens[i - jump + 1:]))
    return ngrams

def timestamp_msec():
    ts = int(round(time.time() * 1000))
    return ts

def handle_exc(excinfo, to_stderr=False):
    s = ''.join(traceback.format_exception(*excinfo))
    if to_stderr:
        log('Error : %s' % s)
    logging.error('Error : %s' % s)

UNFULFILLED_WORDS_PREFIX = "[[ "
UNFULFILLED_WORDS_SUFFIX = " ]]"


class SpotifyTrack(object):
    MAX_NGRAM_SIZE = 15
    def __init__(self, track_ids, title, artist, album, search_terms=None, pop=0):
        self.track_ids = track_ids
        self.title  = remove_non_ascii(title)
        self.artist = remove_non_ascii(artist)
        self.album  = remove_non_ascii(album)
        self.pop = pop
        self.search_tuples = set()
        if search_terms:
            self.add_search_tuple(search_terms)
        tokens = remove_non_ascii(title.lower()).split(' ')
        
        #tokens out of title
        self.tokens = []
        #keep only numbers and letters in tokens
        for token in tokens:
            token = normalize_str(token)
            if len(token.strip()):
                self.tokens.append(token)
            
        #ngrams out of tokens
        self.ngrams = []
        ngramsize = 2
        overlap = 1
        upto = len(tokens) + 1
        if upto > SpotifyTrack.MAX_NGRAM_SIZE:
            upto = SpotifyTrack.MAX_NGRAM_SIZE
        while ngramsize < upto:
            ngrams = ngrams_from_tokens(self.tokens, ngramsize, overlap)
            self.ngrams.extend(ngrams)
            ngramsize += 1
            overlap += 1
        #done

    def add_search_tuple(self, search_terms):
        self.search_tuples.add(search_terms)

    def pretty_print(self, gap=40):
        #assume 128 char gap
        gap1 = gap - len(self.title)
        if gap1 <= 0:
            gap1 = 4
        gap2 = gap - len(self.artist)
        if gap2 <= 0:
            gap2 = 4
        return "    %s%s%s%s%s"  % (self.title,"".join([" "] * gap1), self.artist,"".join([" "] * gap2), self.album)
        
    def debug_print(self):
        return "<"+self.title+"|"+self.artist+"|"+self.album+">"
        
    def __repr__(self):
        return self.pretty_print()
    
    def serialize(self):
        obj = {"ids": self.track_ids, "title": self.title, "artist" : self.artist, "album" : self.album, "search_terms": [[t for t in tup] for tup in self.search_tuples], "pop" : self.pop }
        return json.dumps(obj)
    
    @staticmethod    
    def deserialize(s):
        obj = json.loads(s) 
        track_ids = obj["ids"]
        title = obj["title"]
        artist = obj["artist"]
        album = obj["album"]
        search_terms = [tuple(terms) for terms in obj["search_terms"]]
        pop = obj["pop"]
        spottrack = SpotifyTrack(track_ids, title, artist, album, None, pop)
        for terms in search_terms:
            spottrack.add_search_tuple(terms)
        return spottrack

# This object is used to generate set covers for tokens in a line
class TrackInfoLite(object):
    def __init__(self, first_word, last_word, title_len, track):
        self.first_word = first_word
        self.last_word = last_word
        self.title_len = title_len
        self.tokens = track.tokens
        self.track = track
        
    def __repr__(self):
        #return "<%s:%s:%d:[%s]:%s>" % (self.first_word, self.last_word, self.title_len, " ".join(self.tokens), ",".join(self.track.track_ids))
        return "<'%s':%s>" % (" ".join(self.tokens), ",".join(self.track.track_ids))

class SpotifyAPIClient(object):
    MAX_QUERIES_PER_SEC = 10
    MAX_CONCURRENT_THREADS = 10
    QUEUE_WAIT_TIMEOUT_SEC = 0.2
    def __init__(self):
        self.ngrams_queried = set()
        self.cache = []
        self.trackid_track_map = {}
        self.word_first_last_len_index = {}
        self.title_track_index = {}
        self.ngram_track_index = {}
        self.cache_ngram_wise = False   #set True in case we want granular caching
        #self.word_track_index = {}
        #self.query_track_index = {}
        self.search_track_index = {}
        self.limit_per_sec = 10
        self.track_cache_dirty = False
        self.query_cache_dirty = False
        self.exact_matches = {}
        self.reset()
        self.verbose = False

    def reset(self):
        self.num_requests = 0
        self.last_query_time = 0
        self.queries_waiting = collections.deque()
        self.query_timestamps = collections.deque(maxlen=20)
        self.queries = None
        self.threads = []
        self.responses = Queue.Queue()
        self.tracks = None
        self.nonexact_matches = []

    def init_for_batch_requests(self, batch_size):
        self.reset()
        self.num_requests = batch_size
        self.start_threads(batch_size)
    
    def load_caches(self, fname_prefix, fname_suffix):
        self.load_track_cache(fname_prefix+'_track.'+fname_suffix)
        self.load_query_cache(fname_prefix+'_query.'+fname_suffix)

    def dump_caches(self, fname_prefix, fname_suffix):
        self.dump_track_cache(fname_prefix+'_track.'+fname_suffix)
        self.dump_query_cache(fname_prefix+'_query.'+fname_suffix)

    def load_track_cache(self, fname):
        if not os.path.exists(fname):
            return
        f = open(fname, 'r')
        lines = f.readlines()
        f.close()
        #parse and index tracks
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            spottrack = SpotifyTrack.deserialize(line)
            self.index_track(spottrack)
        self.track_cache_dirty = False

    def dump_track_cache(self, fname):
        #log("dumping to "+fname+" dirty="+str(self.track_cache_dirty))
        if self.track_cache_dirty:
            f = open(fname, 'w')
            #parse and index tracks
            for spottrack in self.cache:
                f.write(spottrack.serialize()+'\n')
            f.close()
            self.track_cache_dirty = False

    def load_query_cache(self, fname):
        if not os.path.exists(fname):
            return
        f = open(fname, 'r')
        lines = f.readlines()
        f.close()
        #parse and index tracks
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            tokens = line.split(',')
            ngrams_tup = tuple(tokens)
            self.ngrams_queried.add(ngrams_tup)
        self.query_cache_dirty = False

    def dump_query_cache(self, fname):
        #log("dumping to "+fname+" dirty="+str(self.query_cache_dirty))
        if self.query_cache_dirty:
            f = open(fname, 'w')
            for ngram_tup in self.ngrams_queried:
                f.write(','.join(list(ngram_tup))+'\n')
            f.close()
            self.query_cache_dirty = False

    def start_threads(self, nqueriers=9):
        nthreads = nqueriers if nqueriers < SpotifyAPIClient.MAX_CONCURRENT_THREADS else SpotifyAPIClient.MAX_CONCURRENT_THREADS #leave one for tolerance?
        self.queries = Queue.Queue(maxsize=nthreads)

        self.tracks = []
        t = threading.Thread(target=self.response_handler, args=())
        t.daemon = True
        t.start()
        self.threads.append(t)
    
        for i in range(nthreads):
            t = threading.Thread(target=self.query_runner, args=(i,))
            t.daemon = True
            t.start()
            self.threads.append(t)
        #self.queue.join()   #block

    #wait on queries to get responses and responses to get parsed        
    def wait_threads(self):
        log("Started threads, waiting on queries and responses")
        self.queries.join()
        #log("Queries done")
        self.responses.join()
        #log("Responses done")
        for t in self.threads:
            t.join()    # this is required in case interpreter shuts down before threads
        log("Queries done, responses parsed")

    def index_track(self, spottrack):
        title_gram = tuple(spottrack.tokens)
        put_as_list(self.title_track_index, title_gram, spottrack)
        first_word = spottrack.tokens[0]
        last_word  = spottrack.tokens[-1]
        title_len  = len(spottrack.tokens)
        put_as_list(self.word_first_last_len_index, first_word, TrackInfoLite(first_word, last_word, title_len, spottrack))
        
        for search_term in spottrack.search_tuples:
            put_as_list(self.search_track_index, search_term, spottrack)
            if self.cache_ngram_wise: 
                put_as_set(self.ngram_track_index, tuple(search_term), spottrack)
        
        if self.cache_ngram_wise: 
            for ngram in spottrack.ngrams:
                put_as_set(self.ngram_track_index, tuple(ngram), spottrack)

        for track_id in spottrack.track_ids:
            self.trackid_track_map[track_id] = spottrack

        #cache
        self.cache.append(spottrack)
        self.track_cache_dirty = True

    def virtually_uncrackable(self, data):
        # 1-time pads are proven impossible to crack
        # http://en.wikipedia.org/wiki/One-time_pad
        otp = "RANDOM_STRING"
        
        # xor this with data in blocks

    def rate_limited_query(self, ngrams_tup):
        if ngrams_tup in self.ngrams_queried or (ngrams_tup in self.title_track_index): 
            #already queried, reuse previous results
            
            cached = self.title_track_index.get(ngrams_tup, set())   #could be empty, doesn't matter (for consistency)
            self.responses.put((ngrams_tup, cached, True))
            #print "Already cached, so NOT queueing query for ngrams_tup'", ngrams_tup, "' cached=", cached
            return

        #print "queueing query for ngrams_tup", ngrams_tup
        self.queries_waiting.append(ngrams_tup) #this is to prevent threads from terminating while we wait for rate limit to expire
        
        self.ngrams_queried.add(ngrams_tup)
        self.query_cache_dirty = True
        
        now_msec = timestamp_msec()
        one_sec_ago = now_msec - 1000
        #find queries in last sec
        #one_sec_tstamp = bisect.bisect_left(self.query_timestamps, one_sec_ago)

        #wait til we have < 10q/sec, and then remove all queries more than a second old
        earliest = 0
        while earliest < (now_msec - 1000) and len(self.query_timestamps):
            earliest = self.query_timestamps.popleft()
            if earliest >= one_sec_ago:
                self.query_timestamps.append(earliest)
                break

        #wait til query rate limit has passed
        while len(self.query_timestamps) >= SpotifyAPIClient.MAX_QUERIES_PER_SEC:
            earliest = self.query_timestamps.popleft()
            if earliest >= (timestamp_msec() - 1000):
                self.query_timestamps.append(earliest)
                time.sleep(0.1)
        
        #start a thread off on it
        now_msec = timestamp_msec()
        self.query_timestamps.append(now_msec)
        self.queries.put(ngrams_tup)
        self.queries_waiting.pop()
        #logging.info("queued query for ngrams_tup %s at t = %u" % (str(ngrams_tup), now_msec))
        
    def query_runner(self, threadnum):
        while True:
            try:
                #get ngrams
                ngrams_tup = self.queries.get(True, SpotifyAPIClient.QUEUE_WAIT_TIMEOUT_SEC)
                
                #make query url
                qstr = "q="+'+'.join([urllib.quote_plus(ngram) for ngram in ngrams_tup]) #urllib.urlencode({'q': '+'.join(ngrams_tup)})
                host = "http://ws.spotify.com/search/1/track.json?"+qstr

                success = False
                
                #run query request
                logging.info("Thread #%d:" % (threadnum)+" Running url fetch: "+host)
                response = None
                try:                
                    url = urllib2.urlopen(host)
                    resp_code = url.getcode()

                    last_mod_ts = 0    #For cache expiry
                    if 'last-modified' in url.headers:
                        last_mod_date = url.headers['last-modified']
                        #when = datetime.datetime.strptime(last_mod_date, "%a, %d %b %Y %H:%M:%S %Z")
                        #last_mod_ts = long(time.mktime(when.timetuple())*1000 + (when.microsecond/1000))
                        #TODO: use this to expire cached query results
                    
                    if resp_code == 200:
                        response = url.read()
                        #log(response[:10] +" --- "+response[100:200])
                        #logging.info("Thread #%d:" % (threadnum)+" URL="+host+" JSON="+response)
                        success = True  
                    elif resp_code == 403:    #rate limit exceeded, sleep to slow down
                        log("EXCEEDED RATE LIMIT!")
                        time.sleep(10)

                    url.close()
                except urllib2.URLError:
                    excinfo = sys.exc_info()
                    handle_exc(excinfo)

                if success:
                    #parse result
                    logging.info("Thread #%d: Handing off to response handler" % (threadnum))
                    self.responses.put((ngrams_tup, response, False))
                else:   #re-queue this one for retrying
                    logging.error("Thread #%d: Error on query " % (threadnum) + str(ngrams_tup)+" at url "+host+", re-queueing")
                    self.rate_limited_query(ngrams_tup)
    
                self.queries.task_done()
            except Queue.Empty:
                if self.queries.empty() and len(self.queries_waiting) < 1:
                    break           # No more jobs in the queue
            except:
                logging.error("Thread #%d:" % (threadnum)+" URL="+host+" response="+response)
                handle_exc(sys.exc_info())

        logging.info("URL fetcher thread #%d done" % (threadnum))

    def response_handler(self):
        num_processed = 0
        while True:
            try:
                #get ngrams
                (ngrams_tup, response, is_parsed) = self.responses.get(True, SpotifyAPIClient.QUEUE_WAIT_TIMEOUT_SEC)
                
                #log("Handling response: "+str(ngrams)+" / "+str(is_parsed)+" / "+str(response[100:200] if not is_parsed else response))
                exact_match = None
                if not is_parsed:
                    #parse
                    results = json.loads(response)
                    
                    #extract tracks
                    trackobjs = results["tracks"]
                    for trackobj in trackobjs:
                        skip = False
                        track_ids = []
                        try :
                            trackidobjs = trackobj["external-ids"]
                            for trackidobj in trackidobjs:
                                trackid = "%s/%s" % (trackidobj["type"], trackidobj["id"])
                                #if already fetched, drop it
                                if trackid in self.trackid_track_map:
                                    spottrack = self.trackid_track_map[trackid]
                                    spottrack.add_search_tuple(ngrams_tup)
                                    if self.cache_ngram_wise:                                     
                                        put_as_set(self.ngram_track_index, ngrams_tup, spottrack)
                                    skip = True
                                    break
                                track_ids.append(trackid)
                        except KeyError:
                            logging.warn("No external-ids in "+json.dumps(trackobj))
                            #make one up randomly
                            base = math.pow(10, 11)
                            while True:
                                rtrackid = 'rndm/%d' %(base + random.randint(1, base/10))
                                if rtrackid not in self.trackid_track_map:
                                    track_ids.append(rtrackid)
                                    break

                        if not skip:
                            albumobj = trackobj['album']
                            album = albumobj['name']
                            title = trackobj['name']
                            pop = trackobj['popularity']
                            
                            artistobjs = trackobj['artists']
                            artists_names = []
                            for artistobj in artistobjs:
                                artists_names.append(artistobj['name'])
                                
                            artist = ', '.join(artists_names)
                            
                            spottrack = SpotifyTrack(track_ids, title, artist, album, ngrams_tup, pop)
                        
                            #index
                            self.index_track(spottrack)

                        tup_tokens = tuple(spottrack.tokens)
                        if tup_tokens == ngrams_tup:
                            exact_match = spottrack
                            put_as_list(self.exact_matches, ngrams_tup, spottrack)
                else:
                    cached_tracks = response
                    best_match = response
                    for spottrack in cached_tracks:
                        tup_tokens = tuple(spottrack.tokens)
                        if tup_tokens == ngrams_tup:
                            exact_match = spottrack
                            put_as_list(self.exact_matches, ngrams_tup, spottrack)                
                num_processed += 1
                
                if exact_match is None:
                    self.nonexact_matches.append(ngrams_tup)
                    #self.tracks.append(best_match)
                
                self.responses.task_done()
            except Queue.Empty:
                #if self.responses.empty() and self.queries.empty() and len(self.queries_waiting) < 1 and (num_processed == self.num_requests):
                if (num_processed == self.num_requests):
                    break           # No more jobs in the queue
            except:
                handle_exc(sys.exc_info())

        logging.info("Response handler thread done, remaining queries=%d responses=%d" %(self.queries.qsize(), self.responses.qsize()))

global_client_ref = None

def get_track_combos_rec(cli, combos, word_stack, tracks, missing_stack, tokens, start, end):
    if len(tokens) < 0:
        combos.append((list(word_stack), list(missing_stack)))
        return
    word_stack.append(tokens[0])
    tokens = tokens[1:]
    pass

def get_unused_if_possible(used, found):
    track = found[0]
    if track in used and len(found) > 0:
        for i in xrange(1, len(found)):
            track = found[i]
            if track not in used:
                break
        #else use whatever we have
        used.add(track)
    return track
            
def get_track_combos_greedy(cli, used_tracks, line, tokens):
    result = []
    nmissing_words = 0
    line_tup = tuple(tokens)
    line_exact_matches = cli.exact_matches.get(line_tup, None)
    if line_exact_matches:
        track = get_unused_if_possible(used_tracks, line_exact_matches)
        result.append(track)
    else:
        start = 0
        end = 1
        #Try greedy approach first
        last_found_start = 0
        last_found_end = 0
        missing_words = []
        while start <= (len(tokens) - 1):
            end = start + len(tokens)
            if end > len(tokens):
                end = len(tokens)
            if start == end:
                break
            #get the longest sequence of tokens that match a track title
            #print "\tlooking up", tokens[start : end], "in map ", len(cli.title_track_index)
            found = None
            while end > (start + 1):
                ngram_tup = tuple(tokens[start : end])
                #print "\tlooking up", ngram_tup
                if ngram_tup in cli.title_track_index:
                    found = cli.title_track_index[ngram_tup]
                    #print "Found!", len(found)
                    last_found_start = end
                    last_found_end = len(tokens)
                    break
                end -= 1
                #sys.exit(1)
            if found is None and end - start == 1:    # no progress at all??
                #check if the word occurs anywhere
                word = tokens[start]
                missing_words.append(word)
                nmissing_words += 1
            else:
                #deal with any previous missing words by collapsing them into a phrase
                if len(missing_words):
                    #print "Not found at all:", missing_words
                    result.append(UNFULFILLED_WORDS_PREFIX+" ".join(missing_words)+UNFULFILLED_WORDS_SUFFIX) #append the word as it is
                    missing_words = []
                #get unused track (try getting most popular?)
                track = get_unused_if_possible(used_tracks, found)
                result.append(track)
            start = end

        if len(missing_words):
            #print "Not found at all:", missing_words
            result.append(UNFULFILLED_WORDS_PREFIX+" ".join(missing_words)+UNFULFILLED_WORDS_SUFFIX) #append the word as it is
            missing_words = []
        
    return result, nmissing_words
    
#Combinatorial, but should not be a problem for N on the order of number of words in lines of poems
def get_covering_indices(ntokens, index_tups, stack, ret):
    if len(index_tups) < 1:
        #calculate missing words
        start = 0
        nmissing = 0
        missing = []
        combo = list(stack) #sorted(stack, key=lambda x: x[0])
        for tup in combo:
            if start < tup[0]:
                #nmissing += (tup[0] - start)
                missing.extend(range(start, tup[0]))
            start = tup[1] + 1
        #nmissing += (ntokens - 1) - tup[1] #trailing misses
        missing.extend(range(start, ntokens))
        nmissing = len(missing)
        #print "*** Result[",ntokens,"]", combo, missing, nmissing
        ret.append((combo, missing))
        return
    remaining = set(index_tups)
    for index_tup in index_tups:
        stack.append(index_tup)
        greater = [next_tup for next_tup in index_tups if next_tup[0] > index_tup[1]]
        #print "\t\t", stack, ": ", index_tup, " < ", greater
        get_covering_indices(ntokens, greater, stack, ret)
        stack.pop()
    return
    
    
def get_track_combos_opt(cli, used_tracks, line, tokens):
    result = []
    nmissing_words = 0
    line_tup = tuple(tokens)
    line_exact_matches = cli.exact_matches.get(line_tup, None)
    if line_exact_matches:
        track = get_unused_if_possible(used_tracks, line_exact_matches)
        result.append(track)
    else:
        start = 0
        end = 1
        #Find all tracks beginning with this word, and check which fit this line
        #Try greedy approach first
        ntokens = len(tokens)
        index_combos_map = {}
        indexes = set()
        while start < ntokens:
            word = tokens[start]
            track_infos_from_word = cli.word_first_last_len_index.get(word, [])
            for track_info in track_infos_from_word:
                ltitle = track_info.title_len
                if start + ltitle > ntokens:
                    continue    #no use to us
                end = start + ltitle - 1
                if tokens[end] == track_info.last_word and (tokens[start:end + 1] == track_info.tokens):
                    index_tup = (start, end)
                    put_as_list(index_combos_map, index_tup, track_info)
            start += 1
        
        #print line, ": Combos\n", "\n".join([str(k)+":"+str(v) for (k, v) in index_combos_map.iteritems()])

        #get covering combos with minimum missing words
        #print "\nLINE:", line
        ret = []
        stack = []
        get_covering_indices(ntokens, index_combos_map.keys(), stack, ret)
        
        if len(ret) > 0:
            #sort by missing
            ret.sort(key = lambda x: len(x[1]))
            min_missing = len(ret[0][1])
            #print "\tSorted min missing: ", ret, min_missing
                        
            #filter to remove anything more than minimum missing words
            #print "\tSorted: ", ret
            ret = [r for r in ret if len(r[1]) == min_missing]
            #print "\tNot min: ", ret
            
            #sort to get one with minimum combos
            ret.sort(key = lambda x: len(x[0]))
            least = len(ret[0][0])
            #print "\tLeast: ", least
            ret = [r for r in ret if len(r[0]) == least]
            #print "\tLeast not min: ", ret
            
            #choose the ones that are unused
            max_unused = -1
            max_unused_combo = []
            missing = []
            for index_tups, index_missing in ret:
                n_unused = 0
                for index_tup in index_tups:
                    combo_track_infos = index_combos_map[index_tup]
                    for combo_track_info in combo_track_infos:
                        if combo_track_info.track not in used_tracks:
                            n_unused+=1
                if n_unused > max_unused:
                    max_unused = n_unused
                    max_unused_combo = index_tups
                    missing = index_missing
            
            #ret = [(index_combos_map[ix], missing, len([track_info for track_info in index_combos_map[ix] if track_info.track not in used_tracks])) for ix in ixs for (ixs, missing) in ret]
            hit_tups = [(True, tup) for tup in max_unused_combo] 
            #print "Least not min most unused:",hit_tups, "/", missing
            nmissing_words = len(missing)
        else:
            #all missing!
            hit_tups = []
            missing = tokens
            
        #print "Hits/Misses", hit_tups, "/", missing
        #group up missing tokens
        missing_tups = []            
        if len(missing):
            start = end = missing[0]
            for i in xrange(1, len(missing)):
                if (missing[i] - missing[i - 1]) == 1:
                    end = missing[i]
                else:
                    missing_tups.append((False, (start, end + 1)))
                    start = end = missing[i]
            missing_tups.append((False, (start, end + 1)))

        #mix ret and missing
        mixed = list(hit_tups) + missing_tups

        ##print "Mixed", mixed
        mixed.sort(key=lambda x: x[1][0])
        
        result = []
        for ix in mixed:
            tup = ix[1]
            if ix[0]:
                result.append(get_unused_if_possible(used_tracks, [track_info.track for track_info in index_combos_map[tup]]))
            else:
                result.append(UNFULFILLED_WORDS_PREFIX+" ".join(tokens[tup[0] : tup[1]])+UNFULFILLED_WORDS_SUFFIX)
        #print "RESULT", result
    
    return result, nmissing_words

def spoetify(cli, lines, linewise_tokens, all_tokens):
    logging.info("Spoetifying %d lines with %d tokens"% (len(lines), len(all_tokens)))
    result = []
    used_tracks = set()
    #get_track_combos = get_track_combos_greedy
    get_track_combos = get_track_combos_opt
    for i in range(len(lines)):
        line = lines[i]
        tokens = linewise_tokens[i]
        track_combos, nmissing_words = get_track_combos(cli, used_tracks, line, tokens)
        result.extend(track_combos)
    return result

# Runs search API requests in 3 passes: 
# 1st pass searching for whole lines at a time;
# 2nd pass searching for ngrams of lines at a time; and 
# 3rd pass searching for any unfulfilled/missing phrases
def run(fname, ngramsize=3, overlap=1):
    if overlap >= ngramsize:
       overlap = ngramsize - 1 
    #read
    all_lines = open(fname, 'r').readlines()
    
    #tokenize
    lines = []
    linewise_tokens = []
    all_tokens = []
    for line in all_lines:
        line = line.strip().encode("utf-8")
        if len(line) == 0:
            continue    #skip empty lines
        lines.append(line)
        ltokens = line.split(' ')
        ltokens = [normalize_str(ltoken) for ltoken in ltokens]
        ltokens = [ltoken for ltoken in ltokens if len(ltoken.strip()) > 0]
        all_tokens.extend(ltokens)        
        linewise_tokens.append(ltokens)

    linegrams = set([tuple(ltokens) for ltokens in linewise_tokens])

    #search on ngrams
    client = SpotifyAPIClient()
    global_client_ref = client  # reference in case of exception
    client.load_caches("cache", "json")

    #First pass: search for whole lines at a time
    api_latency_ms = 0.5
    est_time_ms = api_latency_ms * len(linegrams) / 9.0
    log("It will take about %3.2f (or %.2f sec assuming %3.2f sec API latency and 0 cache hits) seconds to query all %d lines."\
         % (len(linegrams)/10.0, est_time_ms, api_latency_ms, len(linegrams)))
    client.init_for_batch_requests(len(linegrams))

    for ngrams_tup in linegrams:
        client.rate_limited_query(ngrams_tup)
    client.wait_threads()
    

    #Second pass: search for ngrams of umatched lines
    unmatched_lines = list(client.nonexact_matches)

    #check any nonexact matches and query for them ngram-wise
    ngrams = ngrams_from_linewise_tokens(unmatched_lines, ngramsize, overlap)


    est_time_ms = api_latency_ms * len(ngrams) / 9.0
    log("It will take about %3.2f seconds (or %3.2f sec assuming %3.2f sec API latency and 0 cache hits) to query all %d %d-grams with %d overlap"\
        % (len(ngrams)/10.0, est_time_ms, api_latency_ms, len(ngrams), ngramsize, overlap))
    client.init_for_batch_requests(len(ngrams))
    for ngrams_tup in ngrams:
        client.rate_limited_query(ngrams_tup)
    client.wait_threads()

    #dump updated cache
    client.dump_caches("cache", "json")

    #make poem out of responses
    poem = spoetify(client, lines, linewise_tokens, all_tokens)

    #Third pass: search for unmatched phrases / words
    str_type = type("str")
    strip_prefix = len(UNFULFILLED_WORDS_PREFIX)
    strip_suffix = len(UNFULFILLED_WORDS_SUFFIX)
    unmatched_lines = [line[strip_prefix:-strip_suffix] for line in poem if type(line) == str_type]
    unmatched_line_indexes = [ix for ix in xrange(len(poem)) if type(poem[ix]) == str_type]

    #One last targeted try to get unmatched lines/phrases
    client.init_for_batch_requests(len(unmatched_lines))

    unmatched_ngrams = []
    unmatched_tokens = []
    for unmatched_line in unmatched_lines:
        ngrams = unmatched_line.split(" ")
        ngrams_tup = tuple(ngrams)
        #unmatched_tokens.append(ngrams)
        unmatched_ngrams.append(ngrams_tup)
        client.rate_limited_query(ngrams_tup)    
    client.wait_threads()
    
    used = set()    #new tracks expected
    #poem2 = spoetify(client, unmatched_lines, unmatched_tokens)    # doesn't work well, needs more tweaking
    for ix in xrange(len(unmatched_lines)):
        found = client.exact_matches.get(unmatched_ngrams[ix], None)
        if found:
            track = get_unused_if_possible(used, found)
            poem[unmatched_line_indexes[ix]] = track

    #dump updated cache
    client.dump_caches("cache", "json")

    print ""
    for line in poem:
        print line

    #done
    
def main(args):
    #init logging
    logging.basicConfig(filename='spoetify.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    fname = args[1]
    ngrams = 3 if len(args) <= 2 else int(args[2])
    overlap = 1 if len(args) <= 3 else int(args[3])
    run(fname, ngrams, overlap)


# Assumptions and Implementation notes in README.txt

if __name__ == '__main__':
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        pass
    except:
        handle_exc(sys.exc_info(), to_stderr=True)
        pass
    finally:
        #write cache to file
        if global_client_ref:
            global_client_ref.dump_caches("cache", "json")
        pass
