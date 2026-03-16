/* graph-for-funcs.scala

   This script returns a JSON representation of the graph resulting from
   combining the AST, CFG, and PDG for each method contained in the loaded CPG.

   Input: A valid CPG
   Output: JSON
 */

import scala.jdk.CollectionConverters._

import io.circe.syntax._
import io.circe.generic.semiauto._
import io.circe.{Encoder, Json}

import io.shiftleft.semanticcpg.language.types.expressions.generalizations.CfgNode
import io.shiftleft.codepropertygraph.generated.EdgeTypes
import io.shiftleft.codepropertygraph.generated.NodeTypes
import io.shiftleft.codepropertygraph.generated.nodes
import io.shiftleft.dataflowengineoss.language._
import io.shiftleft.semanticcpg.language._
import io.shiftleft.semanticcpg.language.types.expressions.Call
import io.shiftleft.semanticcpg.language.types.structure.Local
import io.shiftleft.codepropertygraph.generated.nodes.MethodParameterIn

import overflowdb._
import overflowdb.traversal._

final case class GraphForFuncsFunction(
  function: String,
  file: String,
  id: String,
  AST: List[nodes.AstNode],
  CFG: List[nodes.AstNode],
  PDG: List[nodes.AstNode]
)

final case class GraphForFuncsResult(functions: List[GraphForFuncsFunction])

implicit val encodeEdge: Encoder[OdbEdge] =
  (edge: OdbEdge) =>
    Json.obj(
      ("id", Json.fromString(edge.toString)),
      ("in", Json.fromString(edge.inNode.toString)),
      ("out", Json.fromString(edge.outNode.toString))
    )

implicit val encodeNode: Encoder[nodes.AstNode] =
  (node: nodes.AstNode) =>
    Json.obj(
      ("id", Json.fromString(node.toString)),
      ("edges", Json.fromValues((node.inE("AST", "CFG").l ++ node.outE("AST", "CFG").l).map(_.asJson))),
      ("properties", Json.fromValues(node.propertyMap.asScala.toList.map { case (key, value) =>
        Json.obj(
          ("key", Json.fromString(key)),
          ("value", Json.fromString(value.toString))
        )
      }))
    )

implicit val encodeFuncFunction: Encoder[GraphForFuncsFunction] = deriveEncoder
implicit val encodeFuncResult: Encoder[GraphForFuncsResult] = deriveEncoder

@main def main(): Json = {
  GraphForFuncsResult(
    cpg.method.map { method =>
      val methodName = method.fullName
      val methodId = method.toString
      val methodFile = method.location.filename
      val methodVertex: Vertex = method

      val astChildren = method.astMinusRoot.l
      val cfgChildren = method.out(EdgeTypes.CONTAINS).asScala.collect { case node: nodes.CfgNode => node }.toList

      val local = new NodeSteps(
        methodVertex
          .out(EdgeTypes.CONTAINS)
          .hasLabel(NodeTypes.BLOCK)
          .out(EdgeTypes.AST)
          .hasLabel(NodeTypes.LOCAL)
          .cast[nodes.Local]
      )
      val sink = local.evalType(".*").referencingIdentifiers.dedup
      val source = new NodeSteps(methodVertex.out(EdgeTypes.CONTAINS).hasLabel(NodeTypes.CALL).cast[nodes.Call]).nameNot("<operator>.*").dedup

      val pdgChildren = sink
        .reachableByFlows(source)
        .l
        .flatMap { path =>
          path.elements.map {
            case trackingPoint @ (_: MethodParameterIn) => trackingPoint.start.method.head
            case trackingPoint => trackingPoint.cfgNode
          }
        }
        .filter(_.toString != methodId)

      GraphForFuncsFunction(methodName, methodFile, methodId, astChildren, cfgChildren, pdgChildren.distinct)
    }.l
  ).asJson
}
