using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Text;

namespace DupExtractor;

/// <summary>
/// Turns a site into a token stream that is identical for copies that differ only in naming
/// and cosmetic style:
///   - locals, parameters and range variables become v1, v2, ...; fields become f1, f2, ...;
///     generic type parameters become g1, g2, ...
///     (resolved through the semantic model, with a syntactic fallback)
///   - string/number/char literals become STR / NUM / CHR
///   - `this.` is dropped, `.ConfigureAwait(...)` is dropped
///   - local declaration types (`var` or explicit) become T
///   - `=> expr` becomes `{ return expr ; }` (or `{ expr ; }` for void)
///   - leading guard clauses (ThrowIfNull, `if (...) throw`) are dropped
/// Member names, type names and keywords are kept: they carry the meaning.
/// One instance per site.
/// </summary>
internal sealed class Normalizer
{
    private static readonly string[] GuardPrefixes =
    {
        "ArgumentNullException.ThrowIf", "ArgumentException.ThrowIf", "ArgumentOutOfRangeException.ThrowIf",
        "ObjectDisposedException.ThrowIf", "Guard.", "Ensure.", "Requires.", "Debug.Assert", "Contract.Requires",
    };

    private readonly SemanticModel _model;
    private readonly HashSet<string> _syntacticFields;
    private readonly HashSet<string> _syntacticTypeParams;
    private readonly HashSet<string> _syntacticLocals = new(StringComparer.Ordinal);
    private readonly Dictionary<ISymbol, string> _bySymbol = new(SymbolEqualityComparer.Default);
    private readonly Dictionary<string, string> _byText = new(StringComparer.Ordinal);
    private readonly HashSet<string> _assigned = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _counters = new(StringComparer.Ordinal);
    private readonly List<(TextSpan Span, string? Replacement)> _spans = new();
    private readonly HashSet<int> _emittedSpans = new();
    private readonly List<TextSpan> _guardSpans = new();
    private readonly List<string> _out = new();

    public Normalizer(SemanticModel model, HashSet<string> fieldNames, HashSet<string> typeParameterNames)
    {
        _model = model;
        _syntacticFields = fieldNames;
        _syntacticTypeParams = typeParameterNames;
    }

    /// <summary>Number of leading guard statements removed from a block body.</summary>
    public int GuardCount => _guardSpans.Count;

    public List<string> Run(Site site)
    {
        foreach (var region in Regions(site)) Collect(region);
        if (site.Block != null) MarkGuards(site.Block);

        foreach (var p in site.Prefix) Emit(p);
        if (site.Block != null)
        {
            Emit(site.Block);
        }
        else if (site.Arrow != null)
        {
            _out.Add("{");
            if (!site.IsVoid) _out.Add("return");
            Emit(site.Arrow.Expression);
            _out.Add(";");
            _out.Add("}");
        }
        return _out;
    }

    /// <summary>Statements in the body, not counting blocks or removed guards.</summary>
    public int CountStatements(Site site)
    {
        if (site.Block == null) return site.Arrow != null ? 1 : 0;
        return site.Block.DescendantNodes()
            .OfType<StatementSyntax>()
            .Count(s => s is not BlockSyntax && !_guardSpans.Any(g => g.Contains(s.Span)));
    }

    /// <summary>Raw tokens (no normalization) for exact-copy hashing.</summary>
    public static List<string> RawTokens(Site site)
    {
        var tokens = new List<string>();
        foreach (var p in site.Prefix) tokens.AddRange(p.DescendantTokens().Select(t => t.Text));
        SyntaxNode? body = (SyntaxNode?)site.Block ?? site.Arrow;
        if (body != null) tokens.AddRange(body.DescendantTokens().Select(t => t.Text));
        return tokens;
    }

    public static bool IsGuard(StatementSyntax st)
    {
        switch (st)
        {
            case ExpressionStatementSyntax { Expression: InvocationExpressionSyntax inv }:
            {
                var callee = SiteFactory.Collapse(inv.Expression.ToString()).Replace(" ", "");
                if (callee.StartsWith("System.", StringComparison.Ordinal)) callee = callee["System.".Length..];
                return GuardPrefixes.Any(p => callee.StartsWith(p, StringComparison.Ordinal));
            }
            case IfStatementSyntax { Else: null } ifs:
                return ifs.Statement is ThrowStatementSyntax
                    || (ifs.Statement is BlockSyntax { Statements.Count: 1 } b && b.Statements[0] is ThrowStatementSyntax);
            default:
                return false;
        }
    }

    private static IEnumerable<SyntaxNode> Regions(Site site)
    {
        foreach (var p in site.Prefix) yield return p;
        if (site.Block != null) yield return site.Block;
        else if (site.Arrow != null) yield return site.Arrow;
    }

    private void Collect(SyntaxNode region)
    {
        foreach (var n in region.DescendantNodesAndSelf())
        {
            switch (n)
            {
                case MemberAccessExpressionSyntax ma when ma.Expression is ThisExpressionSyntax:
                    _spans.Add((TextSpan.FromBounds(ma.Expression.SpanStart, ma.OperatorToken.Span.End), null));
                    break;
                case InvocationExpressionSyntax { Expression: MemberAccessExpressionSyntax cma } inv
                    when cma.Name.Identifier.ValueText == "ConfigureAwait":
                    _spans.Add((TextSpan.FromBounds(cma.OperatorToken.SpanStart, inv.Span.End), null));
                    break;
                case VariableDeclarationSyntax vd
                    when vd.Parent is LocalDeclarationStatementSyntax or ForStatementSyntax or UsingStatementSyntax or FixedStatementSyntax:
                    _spans.Add((vd.Type.Span, "T"));
                    foreach (var v in vd.Variables) _syntacticLocals.Add(v.Identifier.ValueText);
                    break;
                case ForEachStatementSyntax fe:
                    _spans.Add((fe.Type.Span, "T"));
                    _syntacticLocals.Add(fe.Identifier.ValueText);
                    break;
                case DeclarationExpressionSyntax de:
                    _spans.Add((de.Type.Span, "T"));
                    break;
                case SingleVariableDesignationSyntax sv:
                    _syntacticLocals.Add(sv.Identifier.ValueText);
                    break;
                case ParameterSyntax ps:
                    _syntacticLocals.Add(ps.Identifier.ValueText);
                    break;
                case CatchDeclarationSyntax cd when cd.Identifier.ValueText.Length > 0:
                    _syntacticLocals.Add(cd.Identifier.ValueText);
                    break;
                case FromClauseSyntax fc:
                    _syntacticLocals.Add(fc.Identifier.ValueText);
                    break;
                case LetClauseSyntax lc:
                    _syntacticLocals.Add(lc.Identifier.ValueText);
                    break;
                case JoinClauseSyntax jc:
                    _syntacticLocals.Add(jc.Identifier.ValueText);
                    if (jc.Into != null) _syntacticLocals.Add(jc.Into.Identifier.ValueText);
                    break;
                case QueryContinuationSyntax qc:
                    _syntacticLocals.Add(qc.Identifier.ValueText);
                    break;
            }
        }
    }

    private void MarkGuards(BlockSyntax block)
    {
        foreach (var st in block.Statements)
        {
            if (!IsGuard(st)) break;
            _guardSpans.Add(st.Span);
            _spans.Add((st.Span, null));
        }
    }

    private void Emit(SyntaxNode region)
    {
        foreach (var tok in region.DescendantTokens())
        {
            if (tok.IsMissing) continue;
            int idx = OutermostSpan(tok.Span);
            if (idx >= 0)
            {
                var replacement = _spans[idx].Replacement;
                if (replacement != null && _emittedSpans.Add(idx)) _out.Add(replacement);
                continue;
            }
            _out.Add(NormalizeToken(tok));
        }
    }

    private int OutermostSpan(TextSpan s)
    {
        int best = -1, bestLength = -1;
        for (int i = 0; i < _spans.Count; i++)
        {
            var span = _spans[i].Span;
            if (span.Contains(s) && span.Length > bestLength)
            {
                best = i;
                bestLength = span.Length;
            }
        }
        return best;
    }

    private string NormalizeToken(SyntaxToken tok)
    {
        switch (tok.Kind())
        {
            case SyntaxKind.StringLiteralToken:
            case SyntaxKind.InterpolatedStringTextToken:
            case SyntaxKind.SingleLineRawStringLiteralToken:
            case SyntaxKind.MultiLineRawStringLiteralToken:
            case SyntaxKind.Utf8StringLiteralToken:
            case SyntaxKind.Utf8SingleLineRawStringLiteralToken:
            case SyntaxKind.Utf8MultiLineRawStringLiteralToken:
                return "STR";
            case SyntaxKind.NumericLiteralToken:
                return "NUM";
            case SyntaxKind.CharacterLiteralToken:
                return "CHR";
            case SyntaxKind.IdentifierToken:
                return NormalizeIdentifier(tok);
            default:
                return tok.Text;
        }
    }

    private string NormalizeIdentifier(SyntaxToken tok)
    {
        string text = tok.ValueText;
        switch (tok.Parent)
        {
            case IdentifierNameSyntax idn:
            {
                // x.Name, x?.Name, Namespace.Type, named arguments: the name is a member/type/param of someone else.
                if (idn.Parent is MemberAccessExpressionSyntax ma && ma.Name == idn && ma.Expression is not ThisExpressionSyntax) return text;
                if (idn.Parent is MemberBindingExpressionSyntax or QualifiedNameSyntax or NameColonSyntax or AliasQualifiedNameSyntax) return text;

                var sym = ResolvedSymbol(idn);
                switch (sym)
                {
                    case ILocalSymbol or IParameterSymbol or IRangeVariableSymbol:
                        return Placeholder(sym, text, "v");
                    case IFieldSymbol { IsConst: false }:
                        return Placeholder(sym, text, "f");
                    case ITypeParameterSymbol:
                        return Placeholder(sym, text, "g");
                    case null:
                        if (_syntacticLocals.Contains(text)) return Placeholder(null, text, "v");
                        if (_syntacticFields.Contains(text)) return Placeholder(null, text, "f");
                        if (_syntacticTypeParams.Contains(text)) return Placeholder(null, text, "g");
                        return text;
                    default:
                        return text;
                }
            }
            case VariableDeclaratorSyntax or ParameterSyntax or ForEachStatementSyntax or CatchDeclarationSyntax
                or SingleVariableDesignationSyntax or FromClauseSyntax or LetClauseSyntax or JoinClauseSyntax
                or JoinIntoClauseSyntax or QueryContinuationSyntax:
                return Placeholder(DeclaredSymbol(tok.Parent!), text, "v");
            default:
                return text;
        }
    }

    private ISymbol? ResolvedSymbol(IdentifierNameSyntax idn)
    {
        try
        {
            var info = _model.GetSymbolInfo(idn);
            return info.Symbol ?? (info.CandidateSymbols.Length > 0 ? info.CandidateSymbols[0] : null);
        }
        catch { return null; }
    }

    private ISymbol? DeclaredSymbol(SyntaxNode node)
    {
        try { return _model.GetDeclaredSymbol(node); }
        catch { return null; }
    }

    private string Placeholder(ISymbol? sym, string text, string prefix)
    {
        string textKey = prefix + ":" + text;
        if (sym != null && _bySymbol.TryGetValue(sym, out var p)) return p;
        if (_byText.TryGetValue(textKey, out p))
        {
            // Same name seen before. Reuse its placeholder unless it already belongs to a different symbol
            // (a shadowing variable with the same name in another scope).
            if (sym == null) return p;
            if (!_assigned.Contains(p)) { _bySymbol[sym] = p; _assigned.Add(p); return p; }
        }
        int n = _counters.GetValueOrDefault(prefix) + 1;
        _counters[prefix] = n;
        p = prefix + n;
        if (sym != null) { _bySymbol[sym] = p; _assigned.Add(p); }
        _byText.TryAdd(textKey, p);
        return p;
    }
}
