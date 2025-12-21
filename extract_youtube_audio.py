#!/usr/bin/env python3
import os
import re
import sys
import subprocess
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

YOUTUBE_ID_RE = re.compile(r'(?:youtube\.com/(?:watch\?v=|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})')


def fetch_html(url, timeout=15):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extract_video_ids_from_url(url):
    m = YOUTUBE_ID_RE.search(url)
    return m.group(1) if m else None


def extract_youtube_links(html, base_url=None):
    soup = BeautifulSoup(html, 'html.parser')
    ids = []

    # <a> tags
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if 'youtube' in href or 'youtu.be' in href:
            full = urljoin(base_url, href) if base_url else href
            vid = extract_video_ids_from_url(full)
            if vid:
                ids.append(vid)

    # iframe embeds
    for iframe in soup.find_all('iframe', src=True):
        src = iframe['src'].strip()
        if 'youtube' in src or 'youtu.be' in src:
            full = urljoin(base_url, src) if base_url else src
            vid = extract_video_ids_from_url(full)
            if vid:
                ids.append(vid)

    # search raw text
    for m in YOUTUBE_ID_RE.finditer(soup.get_text()):
        ids.append(m.group(1))

    # unique preserving order
    seen = set()
    ordered = []
    for vid in ids:
        if vid not in seen:
            seen.add(vid)
            ordered.append(vid)
    urls = [f'https://www.youtube.com/watch?v={vid}' for vid in ordered]
    return urls


def download_audio_with_yt_dlp(urls, out_dir, audio_format='wav'):
    # build base command
    cmd = ['yt-dlp', '-x', '--audio-format', audio_format]
    outtmpl = os.path.join(out_dir, '%(id)s.%(ext)s')
    cmd += ['-o', outtmpl]

    # If user wants WAV, force sample rate 44100 and 16-bit PCM via ffmpeg args
    if audio_format == 'wav':
        ff_args = '-ar 44100 -sample_fmt s16'
        cmd += ['--postprocessor-args', ff_args]

    cmd += urls
    print('Rulează:', ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print('Eroare: `yt-dlp` nu a fost găsit. Instalați-l (pip install yt-dlp) și încercați din nou.', file=sys.stderr)
        sys.exit(2)
    except subprocess.CalledProcessError as e:
        print('yt-dlp a returnat o eroare:', e, file=sys.stderr)
        sys.exit(e.returncode)


def prompt_input(prompt_text, default=None):
    if default:
        prompt_text = f"{prompt_text} [{default}]: "
    else:
        prompt_text = f"{prompt_text}: "
    val = input(prompt_text).strip()
    return val if val else default


def has_ffmpeg():
    # check ffmpeg and ffprobe are available in PATH
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(['ffprobe', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def validate_ffmpeg_path(path):
    # path can be directory or path to ffmpeg executable
    if not path:
        return False
    path = os.path.abspath(path)
    # if directory, try join
    if os.path.isdir(path):
        ff = os.path.join(path, 'ffmpeg')
        ffprobe = os.path.join(path, 'ffprobe')
        if os.name == 'nt':
            ff += '.exe'
            ffprobe += '.exe'
        return os.path.isfile(ff) and os.path.isfile(ffprobe)
    else:
        # assume user specified executable; check both ffmpeg and ffprobe nearby
        ff = path
        if os.name == 'nt' and not ff.lower().endswith('.exe'):
            ff += '.exe'
        if os.path.isfile(ff):
            # try to find ffprobe in same dir
            d = os.path.dirname(ff)
            ffprobe = os.path.join(d, 'ffprobe')
            if os.name == 'nt':
                ffprobe += '.exe'
            return os.path.isfile(ffprobe)
        return False


def main():
    print('Descărcare audio din videoclipuri YouTube găsite pe o pagină web')
    url = prompt_input('Introduceți URL-ul paginii')
    if not url:
        print('URL invalid. Ieșire.')
        sys.exit(1)

    out_dir = prompt_input('Directorul în care se vor salva fișierele (creează dacă nu există)', default='./out')
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Formatul este fix WAV 44.1kHz 16-bit PCM conform cerinței
    audio_format = 'wav'
    print('Format audio: wav (44.1 kHz, 16-bit PCM)')

    try:
        html = fetch_html(url)
    except Exception as e:
        print('Eroare la descărcarea paginii:', e, file=sys.stderr)
        sys.exit(1)

    links = extract_youtube_links(html, base_url=url)
    if not links:
        print('Nu s-au găsit videoclipuri YouTube pe pagină.')
        sys.exit(0)

    print(f'{len(links)} videoclipuri găsite. Încep descărcarea audio în: {out_dir}')

    # If WAV requested, ensure ffmpeg/ffprobe are available (yt-dlp uses them for conversion)
    ffmpeg_location = None
    if audio_format == 'wav':
        if not has_ffmpeg():
            print('\nffmpeg/ffprobe nu au fost găsite în PATH.')
            print('Trebuie instalate pentru a converti audio la WAV 44.1kHz 16-bit PCM.')
            print('Puteți instala ffmpeg și adăuga în PATH sau introduce calea către directorul care conține ffmpeg/ffprobe.')
            provided = prompt_input('Introduceți calea către directorul ffmpeg (sau lăsați gol pentru a anula)')
            if provided:
                if validate_ffmpeg_path(provided):
                    ffmpeg_location = os.path.abspath(provided)
                    print(f'Folosesc ffmpeg din: {ffmpeg_location}')
                else:
                    print('Calea introdusă nu conține ffmpeg/ffprobe valide. Ieșire.')
                    sys.exit(1)
            else:
                print('Operațiunea a fost anulată. Instalați ffmpeg și reîncercați.')
                print('Instrucțiuni rapide (Windows): https://ffmpeg.org/download.html')
                sys.exit(1)
        else:
            ffmpeg_location = None

    download_cmd_urls = links
    # prepare command addition: if ffmpeg_location provided, pass to yt-dlp
    if ffmpeg_location:
        # pass path to yt-dlp via --ffmpeg-location
        # download_audio_with_yt_dlp will be called with urls; we need to append ffmpeg-location handling here
        # construct command inline to include ffmpeg-location
        cmd = ['yt-dlp', '-x', '--audio-format', audio_format, '-o', os.path.join(out_dir, '%(id)s.%(ext)s')]
        if audio_format == 'wav':
            cmd += ['--postprocessor-args', '-ar 44100 -sample_fmt s16']
        cmd += ['--ffmpeg-location', ffmpeg_location]
        cmd += download_cmd_urls
        print('Rulează:', ' '.join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print('Eroare: `yt-dlp` nu a fost găsit. Instalați-l (pip install yt-dlp) și încercați din nou.', file=sys.stderr)
            sys.exit(2)
        except subprocess.CalledProcessError as e:
            print('yt-dlp a returnat o eroare:', e, file=sys.stderr)
            sys.exit(e.returncode)
    else:
        download_audio_with_yt_dlp(links, out_dir, audio_format=audio_format)
    print('Finalizat.')


if __name__ == '__main__':
    main()
