@echo off

SET SOURCEDIR=source
SET BUILDDIR=build
SET DOCS=..\..\gh-pages

IF /I "%1"=="help" GOTO help
IF /I "%1"=="github" GOTO github
IF /I "%1"=="%" GOTO %
GOTO error

:help
	@%SPHINXBUILD% -M help "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
	GOTO :EOF

:github
	@rm -rf "%BUILDDIR%"
	@make html
	@mv "%BUILDDIR%/html" "%DOCS%"
	@rm -rf "%DOCS%/docs"
	@mv "%DOCS%/html" "%DOCS%/docs"
	@touch "%DOCS%/.nojekyll"
	GOTO :EOF

:%
	CALL make.bat Makefile
	@%SPHINXBUILD% -M $@ "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
	GOTO :EOF

:error
    IF "%1"=="" (
        ECHO make: *** No targets specified and no makefile found.  Stop.
    ) ELSE (
        ECHO make: *** No rule to make target '%1%'. Stop.
    )
    GOTO :EOF
