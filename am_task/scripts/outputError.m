function outputError(datapath, subpath, err)

    cd(datapath);
    fid = fopen('logFile.txt','a+');
    fprintf(fid, '%s', err.getReport('extended', 'hyperlinks','off'));
    fclose(fid);  
    cd(subpath);