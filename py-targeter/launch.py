import papermill
exec(open('./targeter.py').read())
cmd='quarto render targeter-report.qmd --output generated_report  -P tmpdir:"C:\Users\natha\AppData\Local\Temp\targeter_ogarwusp" --to html'
p = subprocess.Popen(cmd, cwd=tmpdir, shell=False)
p.wait()