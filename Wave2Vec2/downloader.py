import youtube_dl

class Downloader:
    class MyLogger(object):
        def debug(self, msg):
            pass
        def warning(self, msg):
            pass
        def error(self, msg):
            print(msg)

    @classmethod
    def download(cls, url, filename):
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': filename,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'logger': cls.MyLogger(),
            'progress_hooks': [cls.my_hook],
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    @staticmethod
    def my_hook(d):
        if d['status'] == 'finished':
            print('Done downloading, now converting ...')