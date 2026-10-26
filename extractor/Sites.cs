using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace DupExtractor;

/// <summary>
/// A method-like piece of code worth comparing: its parameter list / initializer ("prefix")
/// and its body, which is either a block or an expression body.
/// </summary>
internal sealed record Site(
    SyntaxNode Node,
    string Kind,
    string Name,
    string Signature,
    IReadOnlyList<SyntaxNode> Prefix,
    BlockSyntax? Block,
    ArrowExpressionClauseSyntax? Arrow,
    bool IsVoid);

internal static class SiteFactory
{
    private static readonly Regex Ws = new(@"\s+", RegexOptions.Compiled);

    public static string Collapse(string s) => Ws.Replace(s, " ").Trim();

    public static Site? Describe(SyntaxNode node)
    {
        switch (node)
        {
            case BaseMethodDeclarationSyntax m when m.Body != null || m.ExpressionBody != null:
            {
                bool isAsync = m.Modifiers.Any(SyntaxKind.AsyncKeyword);
                string kind, name, sig;
                bool isVoid;
                switch (m)
                {
                    case MethodDeclarationSyntax md:
                        kind = "method";
                        name = md.Identifier.ValueText + (md.TypeParameterList?.ToString() ?? "");
                        isVoid = ReturnsVoid(md.ReturnType, isAsync);
                        sig = $"{md.ReturnType} {name}{md.ParameterList}";
                        break;
                    case ConstructorDeclarationSyntax cd:
                        kind = "constructor";
                        name = cd.Modifiers.Any(SyntaxKind.StaticKeyword) ? ".cctor" : ".ctor";
                        isVoid = true;
                        sig = $"{cd.Identifier.ValueText}{cd.ParameterList}";
                        break;
                    case DestructorDeclarationSyntax dd:
                        kind = "destructor";
                        name = "Finalize";
                        isVoid = true;
                        sig = $"~{dd.Identifier.ValueText}()";
                        break;
                    case OperatorDeclarationSyntax od:
                        kind = "operator";
                        name = "operator " + od.OperatorToken.Text;
                        isVoid = false;
                        sig = $"{od.ReturnType} {name}{od.ParameterList}";
                        break;
                    case ConversionOperatorDeclarationSyntax cv:
                        kind = "operator";
                        name = $"{cv.ImplicitOrExplicitKeyword.Text} operator {cv.Type}";
                        isVoid = false;
                        sig = $"{name}{cv.ParameterList}";
                        break;
                    default:
                        return null;
                }
                var prefix = new List<SyntaxNode> { m.ParameterList };
                if (m is ConstructorDeclarationSyntax { Initializer: not null } ctor) prefix.Add(ctor.Initializer);
                return new Site(m, kind, name, Collapse(sig), prefix, m.Body, m.ExpressionBody, isVoid);
            }

            case LocalFunctionStatementSyntax lf when lf.Body != null || lf.ExpressionBody != null:
            {
                bool isAsync = lf.Modifiers.Any(SyntaxKind.AsyncKeyword);
                var name = lf.Identifier.ValueText + (lf.TypeParameterList?.ToString() ?? "");
                return new Site(lf, "localFunction", name, Collapse($"{lf.ReturnType} {name}{lf.ParameterList}"),
                    new List<SyntaxNode> { lf.ParameterList }, lf.Body, lf.ExpressionBody, ReturnsVoid(lf.ReturnType, isAsync));
            }

            case AccessorDeclarationSyntax a when a.Body != null || a.ExpressionBody != null:
            {
                var owner = a.Parent?.Parent;
                var prefix = new List<SyntaxNode>();
                string ownerName = owner switch
                {
                    PropertyDeclarationSyntax ownerProp => ownerProp.Identifier.ValueText,
                    IndexerDeclarationSyntax => "this[]",
                    EventDeclarationSyntax ownerEvent => ownerEvent.Identifier.ValueText,
                    _ => "?",
                };
                if (owner is IndexerDeclarationSyntax ownerIndexer) prefix.Add(ownerIndexer.ParameterList);
                bool isGetter = a.Keyword.IsKind(SyntaxKind.GetKeyword);
                var name = $"{ownerName}.{a.Keyword.ValueText}";
                return new Site(a, "accessor", name, name, prefix, a.Body, a.ExpressionBody, !isGetter);
            }

            case PropertyDeclarationSyntax p when p.ExpressionBody != null:
                return new Site(p, "accessor", p.Identifier.ValueText + ".get", Collapse($"{p.Type} {p.Identifier.ValueText}"),
                    Array.Empty<SyntaxNode>(), null, p.ExpressionBody, false);

            case IndexerDeclarationSyntax ix when ix.ExpressionBody != null:
                return new Site(ix, "accessor", "this[].get", Collapse($"{ix.Type} this{ix.ParameterList}"),
                    new List<SyntaxNode> { ix.ParameterList }, null, ix.ExpressionBody, false);

            // Only block-bodied lambdas; expression lambdas are almost always too small to matter.
            case AnonymousFunctionExpressionSyntax af when af.Block != null:
            {
                var prefix = new List<SyntaxNode>();
                switch (af)
                {
                    case ParenthesizedLambdaExpressionSyntax pl: prefix.Add(pl.ParameterList); break;
                    case SimpleLambdaExpressionSyntax sl: prefix.Add(sl.Parameter); break;
                    case AnonymousMethodExpressionSyntax { ParameterList: not null } am: prefix.Add(am.ParameterList); break;
                }
                return new Site(af, "lambda", "<lambda>", "<lambda>", prefix, af.Block, null, false);
            }

            default:
                return null;
        }
    }

    private static bool ReturnsVoid(TypeSyntax? type, bool isAsync)
    {
        if (type is null) return true;
        if (type is PredefinedTypeSyntax p && p.Keyword.IsKind(SyntaxKind.VoidKeyword)) return true;
        if (!isAsync) return false;
        var s = type.ToString();
        return s is "Task" or "ValueTask" or "System.Threading.Tasks.Task" or "System.Threading.Tasks.ValueTask";
    }
}
