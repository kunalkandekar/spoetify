                                SPOETIFY
                                ========

Create a list of tracks using Spotify API whose titles constitute the lines of a
given poem.

Examples of Spotify poetry - http://spotifypoetry.tumblr.com

Spotify API homepage - https://developer.spotify.com/technologies/web-api/

(I really should trademark Spoetify :-P)

Usage: 
    python spoetify.py <text-file>

Environment:
    Python version 2.7.2 (Tested only with CPython 2.7.2)
    Any Operating System that runs Python (Tested only on OSX 10.7)

Output:
    List of tracks (if they fulfill a word, phrase or entire line of the poem), 
    or the unfulfilled words wrapped in double square-brackets to differentiate 
    from matching tracks. Track information printed out includes track title,
    the artist(s) name(s) and the album name.

Example:
    <prompt>$ python spoetify.py poem04.txt
        Friday                                  Ice Cube                                At Tha Movies
        Friday                                  Ice Cube                                At Tha Movies
        Friday                                  Ice Cube                                At Tha Movies
        Friday On My Mind                       The Easybeats                           The Definitive Anthology
        Friday I'm In Love                      The Cure                                Wish
        Thank God It's Friday                   R. Kelly                                R. Kelly
        Yeah! Oh, Yeah!                         The Magnetic Fields                     69 Love Songs
        Oh Yeah                                 Bat For Lashes                          The Haunted Man
        Oh Yeah                                 Bat For Lashes                          The Haunted Man
        Bow Chicka Wow Wow                      Mike Posner                             31 Minutes to Takeoff
    [[ fun fun fun ]]
        Ain't It Fun                            Dead Boys                               We Have Come For Your Children
        Say Yes                                 Elliott Smith                           Either Or
    [[ and ]]
        Relax                                   Deep Sleep                              Sleep Music: Lullabies to Help You Relax, Sleep, Meditate and Heal With Relaxing Piano Music, Nature Sounds and Natural Noise

The script logs some progress statements to stderr (just for some basic 
semblance of interactivity.) It also logs some internal steps to a text file
(spoetify.log)


IMPLEMENTATION NOTES / ASSUMPTIONS
# All words in track titles returned by the API are separated by at least one 
  space. This makes parsing much easier, so that not much time is spent there.

# Only English language poems and tracks are expected.

# Punctuation is not important. Pretty much all non-alphanumeric characters 
  (except apostrophes) are considered irrelevant.

# Poem words and/or lines are only satisfied by track titles. So no matching is
  done with the artist and album names.
 
# I try to get all possible relevant tracks up front, and then make the best of
  it rather than returning to query for missing phrases
 
# The script makes search API requests in 3 passes: 
  1st pass searching for whole lines at a time;
  2nd pass searching for ngrams of lines not fulfilled in 1st pass; and 
  3rd pass searching for any unfulfilled/missing phrases in 1st and 2nd passes.
  
  Thus, at each pass, lines / words that are "fulfilled" by track titles are 
  eliminated from future searches. This may result in sub-optimal results, but
  has lower overall latency.
  
  New threads are spawned for each pass, which could certainly be refined, but
  would require a lot more debugging than time permits.
  
# Unfulfilled lines from the 1st pass are broken into overlapping N-grams, and 
  each N-gram is searched separately. By default, the N in N-grams is 3 and the
  overlap between N-grams is 1. They can be changed from command-line arguments,
  but is not recommended.

# API Requests are made in parallel with rate-limiting to avoid going over 10
  requests per second.
 
# Assumed that most relevant tracks are returned on the first page of search 
  results, so not recursing requests to traverse beyond the first page is not 
  done. This trades off potentially more unfulfilled words / phrases in exchange
  for fewer API requests incurred and lower latency. 
  
  This can and does result in unfulfilled words and phrases that do exist as 
  titles for tracks on Spotify, even for the test case poems I tried. But to
  fulfill them would require many more API calls to search deeper into later 
  pages of the results. Thus, I am again trading off fewer API calls for 
  somewhat higher rate of unfulfilled words.
    
# When matching track titles with lines of a poem, preference is given in the 
  following order:
  1. Minimum missing (unfulfilled) words of the poem
  2. Minimum number of tracks required to fulfill each line
  3. Unused tracks, if multiple tracks with the same matching title are found.
  Also, I assumed that repetition of words in the result is not allowed. So 
  overall, preference is given to full coverage of words by track titles as long
  as words don't get repeated, which also results in minimizing the number of 
  tracks.

# The algorithm matching titles with lines is combinatorial in nature. This 
  could get expensive in some cases for lines with many, many, many words, but
  human-written poems don't typically get that long :-P In any case, even for 
  very long lines, the request latency will overwhelm the processing time.

# I am NOT checking if tracks match across lines in a poem (e.g. if the last 
  words of one line and the first words of a subsequent match a track title, I 
  don't consider it.) 
  
  It would be a simple change to enable matching across lines, but I decided 
  against it. I assume poets want to maintain the rhythm and cadence of their 
  poems, and changing line breaks would improperly alter that. After all rhythm
  and cadence are more important to poems than coverage by a matching titles!
  
# I try not to use the same track over and over again if multiple options are 
  available. However, this is only with the tracks available. Additional API
  calls are not incurred in the search for alternate matching tracks.
  
# Queries and retrieved results are cached to files. Cached results are re-used 
  wherever possible to minimize API calls over the network. Cache files are 
  loaded at startup. Thus caching works across sessions. However expiry is not 
  handled yet. Expiring based on last-modified header field of API responses is
  a TODO item.
